classdef MovingWindowPreProcessor < handle
    % MOVINGWINDOEPREPROCESSOR preprocesses data in chunks denoted extracted by a moving window.
    %   Data is fetched from the disk vis the custom `fn_read` function supplied @ the constructor.
    %   Preprocessed data will be written into `tr|val|te` folders under `save_ppdata_dirpath`.
    %   For regression problems: Target.mat file will be created to store the output.  
    
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com).        
    properties (SetAccess = protected)
        params_;            % varargs provided @ the constructor.
    end
    
    properties (SetAccess = private)
        st__;               % Image start index.
        steps__;            % Step size for window.
        wsizes__;           % Window size.
        cur_blk_i__;        % Current on going class index.
        shuffle__;          % Whether to shuffle the training/validation/testing splits.
        splt_ratios__;      % Training/Validation/Testing ratios.
        
        fds__;              % `FileDataStore` object to stream the data in the mat files.
        
        save_dirpath__;     % Save path to write the preprocessed data.
        
        output_matobj__;    % To write the target data on the go. (For incremental writing).
        global_tr_idx__;    % Incremental index used in writing training data to the disk.
        global_val_idx__;   % Incremental index used in writing validation data to the disk.
        global_te_idx__;    % Incremental index used in writing testing data to the disk.
        
        fit_pipelines__;    % Data fit pipelines. 
                            % Each pipeline will invoke a full iteration of the dataset.
        std_pipeline__;     % Standardize pipeline.
    end
    
    properties (Constant)
        PIPELINE_NEW = -1;  % Indicator for a new pipeline.
    end
   
    % For internal use
    properties (Constant)
        BLOCK_TR = 1;       % Indicator for training data block.
        BLOCK_VAL = 2;      % Indicator for validation data block.
        BLOCK_TE = 3;       % Indicator for testing data block.
    end
    
    properties (Dependent)
        IsInitialized       % Is initialized or not.
        HasFitPipelines     % Indicates; fit pipelines are available.
        RegressionMode      % Indicates; operating in regression mode or classification mode.     
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = MovingWindowPreProcessor(location, fn_read, save_ppdata_dirpath, varargin) 
            % Constructs a `MovingWindowPreProcessor` object.
            %
            % Parameters
            % ----------
            % location : string
            %       Path to fetch the data.
            %
            % fn_read : `callable`
            %       Read function that will be utilized to read the data at the path specified by `location`.
            %       Function signature should be as follows:
            %
            %       data = load_(filename), and `data` must be a struct that contain the fields:
            %               data.Input => Format.H_N
            %               data.Ouput => Format.H_N
            %               data.ulbl  -> row vector of unique labels
            %
            % save_ppdata_dirpath : string
            %       Path to save the pre-processed data.
            %
            % varargin : 
            %       FnPreprocess : function         - Custom preprocess function.
            %                                           Refer: MovingWindowPreProcessor.preprocess(...)
            %       DistanceType : string           - Distance type to construct distance matrix
            %                                           i.e 'euclidean', etc. refer `pdist()` for 
            %                                           more details.
            %       SplitRatios : vector -double    - Trianing, validation, testing split ratios.
            %                                           default = [0.7 0.15 0.15]
            %       Shuffle : string                - To shuffle the data after split of training,
            %                                           validation, testing
            %       OutputNodes : int               - For regression problems.
            %                                           default = [], for classification problems
            %       FullDiskCompatibility : bool    - All tables returned will be pointing to the
            %                                           data stored in the disk. default = False
            %       TargetOutputSize : vector -int  - Image target size. (Due to matlab limitation)
            %                                           limitation: ValidationData: No support for datasource
            %                                           This config is used in building the validation table.
            %                                           default = [227 227]
            %
            
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
            import nnf.pp.MovingWindowPreProcessor;

            p = inputParser;
            
            defaultFnDistanceOp = @self.dop_default;
            defaultDistanceType = 'euclidean';
            defaultSplitRatios = [0.7 0.15 0.15];
            defaultShuffle = false;
            defaultOutputNodes = 0;
            defaultFullDiskCompatibility = false;
            defaultTargetImageSize = [];
                        
            addParameter(p, 'FnDistanceOp', defaultFnDistanceOp);
            addParameter(p, 'DistanceType', defaultDistanceType);
            addParameter(p, 'SplitRatios', defaultSplitRatios);
            addParameter(p, 'Shuffle', defaultShuffle);
            addParameter(p, 'OutputNodes', defaultOutputNodes);
            addParameter(p, 'FullDiskCompatibility', defaultFullDiskCompatibility);
            addParameter(p, 'TargetImageSize', defaultTargetImageSize);
            
            parse(p, varargin{:});
            self.params_ = p.Results;
                        
            % Initialize the internal varibles/defaults
            % Preprocess data writing related
            self.save_dirpath__ = save_ppdata_dirpath;
            
            % Query the location and list the files to construct a filestore
            self.fds__ = fileDatastore(location, 'ReadFcn', fn_read, 'FileExtensions','.mat');
            
            % Error handling
            if (self.params_.SplitRatios(2) > 0 && isempty(self.params_.TargetImageSize))
                error(['`TargetImageSize` must be specified for the validation split']);
            end
            
            % Create a directory to save pp data
            [~,~,~] = mkdir(save_ppdata_dirpath);
            
            self.init([], []);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init(self, step_sizes, window_sizes)
            % INIT: Initializes the step sizes and the corresponding window sizes.
            %       Assumtion: (numel(step_sizes) == numel(window_sizes))
            %
            % Parameters
            % ----------
            % step_sizes : vector -int
            %       Step sizes.
            %
            % window_sizes : vector -int
            %       Window sizes.
            %
            
            assert(numel(step_sizes) == numel(window_sizes));
            
            % Window related params
            self.st__ = 1;
            self.steps__ = step_sizes;
            self.wsizes__ = window_sizes;
            self.cur_blk_i__ = 1;
            
            % For regression problems
            if (self.params_.OutputNodes > 0)
                self.output_matobj__ = matfile(fullfile(self.save_dirpath__, 'Target.mat'), 'Writable', true);
                self.output_matobj__.TrOutput = zeros(1, self.params_.OutputNodes);
                self.output_matobj__.ValOutput = zeros(1, self.params_.OutputNodes);
                self.output_matobj__.TeOutput = zeros(1, self.params_.OutputNodes);
            end
            
            % Preprocess data writing related
            self.global_tr_idx__ = 1;
            self.global_val_idx__ = 1;
            self.global_te_idx__ = 1;
            
            self.fit_pipelines__ = {};
            self.std_pipeline__ = {};
            
            % Resets the ongoing indices to read the mat file via `fn_read` provided @ the constructor
            self.reset__();
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pipeline_id = add_to_fit_pipeline(self, pipeline_id, stream_pp)
            % ADD_TO_FIT_PIPELINE: Add `StreamDataPreProcessor` object to specified 
            %       fit-pipeline denoted by `pipeline_id`.
            %       Use `MovingWindowPreProcessor.PIPELINE_NEW` for a new pipeline.
            %
            % Parameters
            % ----------
            % pipeline_id : int
            %       Unique id of the fit-pipeline.
            %
            % stream_pp : :obj:`StreamDataPreProcessor`
            %       Preprocessor object for streaming data.
            %
            % Returns
            % -------
            % pipeline_id : int
            %       Unique id of the fit-pipeline.
            %
            
            % Imports
            import nnf.pp.MovingWindowPreProcessor
            
            if (pipeline_id == MovingWindowPreProcessor.PIPELINE_NEW)
                self.fit_pipelines__{end + 1} = {stream_pp};
                pipeline_id = numel(self.fit_pipelines__);
            else
                pipeline = self.fit_pipelines__{pipeline_id};
                pipeline{end + 1} = stream_pp;
                self.fit_pipelines__{pipeline_id} = pipeline;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function add_to_standardize_pipeline(self, stream_pp)
            % ADD_TO_STANDARDIZE_PIPELINE: Add `StreamDataPreProcessor` object to standardize
            %       pipeline denoted by `pipeline_id`.
            %
            % Parameters
            % ----------
            % stream_pp : :obj:`StreamDataPreProcessor`
            %       Preprocessor object for streaming data.
            %
            self.std_pipeline__{end + 1} = stream_pp;
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [tr_table, val_table, te_table] = preprocess(self)
            % PREPROCESS: Performs moving window preprocessing.
            %            
            % Returns
            % -------
            % tr_table : table
            %       Training table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label for classification problems
            %                           <Empty> for regression problems (data written to the disk)
            %
            % val_table : table
            %       Validation table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label or target output vector
            %
            % te_table : table
            %       Testing table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label or target output vector
            %
            
            % Error handling
            if (~self.IsInitialized)
                error(['Invoke init() method before calling preprocess() method']);
            end
            
            % Fit the data to pre-processing pipelines
            if (self.HasFitPipelines)
                self.process_fit_pipelines__();
            end                     
            
            tr_table = table();            
            val_table = table();
            te_table = table();
            
            debug_i = 4;
            
            % Loop till fds is exhausted
            while hasdata(self.fds__)
                
                % TODO: Remove
                if (debug_i <= 0)
                    break;
                end
                debug_i = debug_i - 1;
                
                % Read the mat file
                mat = self.fds__.read();
                
                % Sample count
                n = size(mat.Input, 2);
                    
                % Fetch start of the boudaries
                [blk_sts, btypes] = self.get_blocks_info__(mat.ulbl);
                
                % Shuffle the data within a class if required
                if (self.params_.Shuffle)
                    mat.Input = self.shuffle_within_cls__(mat.Input, mat.ulbl);
                end
                                                
                % Loop through all windows sizes
                for idx = 1:numel(self.wsizes__)
                    wsize = self.wsizes__(idx);
                    step = floor(self.steps__(idx));
                    prefix = [num2str(wsize) '_' num2str(step)];
                    
                    % Reset iterator for each file
                    self.reset__();
                    
                    % Reset containers for each file
                    targets_per_block = [];
                    inputs_per_block = [];
                    
                    % Move the window of chosen size on the entire dataset
                    while (true)
                        
                        % Image end index
                        en = self.st__ + wsize - 1;
                        
                        % Calculated current block end
                        [cur_blk_en, btype] = self.get_cur_block_en__(blk_sts, btypes, n);
                        
                        % Fix the end
                        if (en > cur_blk_en)
                            % warning(sprintf(['windows size (=%d) with the current offset (=%d) '...'
                            %    'will exceed the current class limit. Truncated data will be returned.'], ...
                            %    wsize, self.st__));
                            en = cur_blk_en;
                        end
                        
                        % Standardize input data
                        input = self.process_standardize_pipeline__(mat.Input(:, self.st__:en), mat.ulbl(self.st__:en));
                        
                        % Perform the distance calculateion
                        input = self.params_.FnDistanceOp(input, wsize);
                        in_count = size(input, 3);
                        inputs_per_block = cat(3, inputs_per_block, input);
                        
                        if (isempty(mat.Output) && self.RegressionMode)
                            error(['`Output` field of the data file is empty.']);
                        end
                        
                        if (~isempty(mat.Output))
                            assert(self.RegressionMode);
                            targets_per_block = cat(2, targets_per_block, ...
                                repmat(mat.Output(:, blk_sts(self.cur_blk_i__)), 1, in_count));
                        end
                        
                        % Last window for this block
                        % Fix image start index `self.st__` and `self.cur_blk_i__`
                        if (en == cur_blk_en)
                            % Before updating the cur_blk_i__, write the data to disk
                            [tr_table, val_table, te_table] = ...
                                self.write_data__(tr_table, val_table, te_table, ...
                                    inputs_per_block, targets_per_block, btype, mat.ulbl(blk_sts(self.cur_blk_i__)), prefix);
                            
                            % Reset containers for each block
                            inputs_per_block = [];
                            targets_per_block = [];
                            
                            self.st__ = en + 1;
                            self.cur_blk_i__ = self.cur_blk_i__ + 1;
                            
                        else
                            % Window is still moving
                            self.st__ = self.st__ + step;                            
                        end
                        
                        % Fix `self.cur_blk_i__` incase new `self.st__` has jumped this block's block due
                        % to a big step size
                        prev_value = self.cur_blk_i__;
                        for i=self.cur_blk_i__:numel(blk_sts)
                            if (blk_sts(i) <= self.st__ && (prev_value < i))
                                
                                % Before updating the cur_blk_i__, write the data to disk
                                [tr_table, val_table, te_table] = ...
                                    self.write_data__(tr_table, val_table, te_table, ...
                                        inputs_per_block, targets_per_block, btype, mat.ulbl(blk_sts(i)), prefix);
                                                            
                                % Reset containers for each block
                                inputs_per_block = [];
                                targets_per_block = [];
                                
                                % Update `cur_blk_i__`
                                self.cur_blk_i__ = i;
                                
                            else
                                break;
                            end
                        end
                        
                        if (en >= n)
                            break;
                        end
                        
                        if (self.st__ > n)
                            break;
                        end
                    end
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function input = dop_default(self, input, wsize)
            % Input data format = Format.H_N
            % Must return a 2D matrix
            
            input = input'; 
            input = squareform(pdist(input, self.params_.DistanceType));
            
            if ~isequal(size(input), [wsize wsize])
                input = imresize(input, [wsize wsize]);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function input = dop_square(self, input, wsize)
            % Input data format = Format.H_N
            % Must return a 2D matrix
            
            if ~isequal(size(input), wsize)
                input = imresize(input, wsize);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function reset__(self)
            % RESET: Resets the ongoing indices to read the mat file via `fn_read` provided @ 
            %           the constructor.
            self.st__ = 1;
            self.cur_blk_i__ = 1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function process_fit_pipelines__(self)
            % PROCESS_FIT_PIPELINES__: Fits the data onto fit pipelines. Invoke `fit()` method of 
            %       `StreamDataPreProcessor` objects in each pipeline.
            %       
            %       IMPORTANT: Each fit-pieline will request full round of data traversal. (PERF hit).
            %
            
            for pipeline_id=1:numel(self.fit_pipelines__)
                pipeline = self.fit_pipelines__{pipeline_id};
                self.fds__.reset();
                
                % Loop till fds is exhausted for each pipeline
                while hasdata(self.fds__)
                    % Read the mat file
                    mat = self.fds__.read();
                    
                    % Fetch all training indices only
                    [~, ~, tr_indices, ~, ~] = self.get_blocks_info__(mat.ulbl);
                    
                    for j=1:numel(pipeline)
                        stream_pp = pipeline{j};
                        stream_pp.fit(pipeline_id, ...
                            ~hasdata(self.fds__), ...
                            mat.Input(:, tr_indices), ...
                            mat.ulbl(tr_indices));
                    end
                end
            end
            
            % Reset the `fds` state
            self.fds__.reset();
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [data_block] = process_standardize_pipeline__(self, data_block, ulbl)
            % PROCESS_STANDARDIZE_PIPELINE__: Invoke `standardize()` method of 
            %       `StreamDataPreProcessor` objects in the standardize pipeline.
            %
            % Parameters
            % ----------
            % data_block : 2D tensor -double
            %       Data tensor in the format: Features x Samples
            %
            % ulbl : vector -int
            %       Unique label vector for the data block.
            %
            % Returns
            % -------
            % data : 2D tensor -double
            %       Standardized data tensor in the format: Features x Samples
            %
            
            for j=1:numel(self.std_pipeline__)
                stream_pp = self.std_pipeline__{j};
                data_block = stream_pp.standardize(data_block, ulbl);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function input = shuffle_within_cls__(self, input, ulbl)
            % TODO:
        end    
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [blk_sts, btypes, tr_indices, val_indices, te_indices] = get_blocks_info__(self, ulbl)
            
            % Imports
            import nnf.pp.MovingWindowPreProcessor;
            import nnf.utl.divideblockex;
            
            % Fetch indices of each class's start
            [~, cls_sts, ~] = unique(ulbl, 'stable');
            
            % Assumption: Samples belongs to same class must lie in consecutive blocks
            assert(isequal(sort(cls_sts), cls_sts));
            
            btypes = [];    % Type of the block (one of the types BLOCK_TR|VAL|TE) 
            blk_sts = [];   % Block starts
            blk_i = 1;      % Block index
            
            % All training indices
            tr_indices = [];
            val_indices = [];
            te_indices = [];
                        
            st = 1;
            cls_i = 1;
            ratios = self.params_.SplitRatios;
            
            % Construct block starts and type of each block
            while (true) 
                % Last element
                if (cls_i + 1 > numel(cls_sts))
                    en = numel(ulbl);
                else
                    en = cls_sts(cls_i + 1) - 1;
                end       
                
                [tmp_tr_indices, tmp_val_indices, tmp_te_indices] = ...
                    divideblockex(st:en, ratios(1), ratios(2), ratios(3));
                st = en + 1;
                tr_indices = cat(2, tr_indices, tmp_tr_indices);
                val_indices = cat(2, val_indices, tmp_val_indices);
                te_indices = cat(2, te_indices, tmp_te_indices);
                
                % Update block starts and types
                if ~isempty(tmp_tr_indices)
                    blk_sts(blk_i) = min(tmp_tr_indices);
                    btypes(blk_i) = MovingWindowPreProcessor.BLOCK_TR;
                    blk_i = blk_i + 1;
                end
                
                if ~isempty(tmp_val_indices)
                    blk_sts(blk_i) = min(tmp_val_indices);
                    btypes(blk_i) = MovingWindowPreProcessor.BLOCK_VAL;
                    blk_i = blk_i + 1;
                end
                
                if ~isempty(tmp_te_indices)
                    blk_sts(blk_i) = min(tmp_te_indices);
                    btypes(blk_i) = MovingWindowPreProcessor.BLOCK_TE;
                    blk_i = blk_i + 1;
                end
                
                cls_i = cls_i + 1;
                
                if (cls_i > numel(cls_sts))
                    break;
                end
            end
                     
        end       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [cur_blk_en, btype] = get_cur_block_en__(self, blk_sts, btypes, n)
            btype = btypes(self.cur_blk_i__);
            
            if (self.cur_blk_i__ < numel(blk_sts))
                cur_blk_en = blk_sts(self.cur_blk_i__ + 1) - 1;
            else
                cur_blk_en = n;
            end
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [tr_table, val_table, te_table] = ...
                write_data__(self, tr_table, val_table, te_table, ...
                                        data_per_block, target_per_block, btype, cls_lbl, prefix)
            % WRITE_DATA__: Write data to the disk depending on `btype` (data block type) and
            %               add entries to `tr_table`, `val_table`, `te_table` as necessary.
            %
            % Parameters
            % ----------
            % tr_table : table
            %       Training table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label for classification problems
            %                           <Empty> for regression problems (data written to the disk)
            %
            % val_table : table
            %       Validation table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label or target output vector
            %
            % te_table : table
            %       Testing table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label or target output vector
            %            
            % data_per_block : 3D tensor -double
            %       Data tensor in the format: H x W x N
            %
            % target_per_block : 2D tensor -double
            %       Target data tensor in the format: Features x Samples
            %            
            % btype : int
            %       Data block type. One of (MovingWindowPreProcessor.BLOCK_TR|VAL|TE).
            %
            % cls_lbl : int
            %       Class label for the `data_per_block`.
            %
            % prefix : string
            %       Prefix to be used when writing data samples to the disk.
            %
            % Returns
            % -------
            % tr_table : table
            %       Augmented training table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label for classification problems
            %                           <Empty> for regression problems (data written to the disk)
            %
            % val_table : table
            %       Augmented validation table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label or target output vector
            %
            % te_table : table
            %       Augmented testing table having links to the image files written to the disk.
            %           1^st Column: Image filepath to disk
            %           2:end^ Columns: Categorical label or target output vector
            %
            
            % Imports
            import nnf.utl.imsave_tiff;
            import nnf.utl.divideblockex;
            import nnf.pp.MovingWindowPreProcessor;
            
            % Create tables to append rows at a time.
            % Matlab does not allow table columns to be appended but rows.
            
            % If `self.st__` jumped two blocks ahead, data_per_block will be empty
            if (isempty(data_per_block))
                return;
            end

            % Normalize the distance into [0 1]
            data_per_block = data_per_block / max(max(max(data_per_block)));
            
            % Write training split
            if (btype == MovingWindowPreProcessor.BLOCK_TR)
                % 2D image dataset (similar to grayscale)
                for idx = 1:size(data_per_block, 3)
                    im = data_per_block(:, :, idx);        % Assumption 2D image

                    % Input to the disk
                    % For regression problems
                    if (self.RegressionMode)
                        cls_lbl = [];
                    end                

                    % Absolute filename to save data
                    subdir = fullfile(self.save_dirpath__, 'tr', num2str(cls_lbl));
                    [~,~,~] = mkdir(subdir);
                    filename = fullfile(subdir, [num2str(self.global_tr_idx__) '_' prefix '_' num2str(idx) '.tiff']);

                    % Write pp data                        
                    imsave_tiff(im, filename);

                    % Build the tr_table
                    if (~self.RegressionMode)
                        row = cell(1, 2);
                        row{1, 1} = filename;
                        row{1, 2} = categorical(cls_lbl);
                        tr_table(self.global_tr_idx__, :) = cell2table(row);
                        self.global_tr_idx__ = self.global_tr_idx__ + 1;
                        continue;   

                    else
                        row = cell(1, 1); 
                        row{1, 1} = filename;
                        tr_table(self.global_tr_idx__, :) = cell2table(row);
                    end

                    % Output to the disk (only for regression problems)
                    self.output_matobj__.TrOutput(self.global_tr_idx__, :) = target_per_block(:, idx)';
                    self.global_tr_idx__ = self.global_tr_idx__ + 1;
                end
            end
            
            % Write validation split
            if (btype == MovingWindowPreProcessor.BLOCK_VAL)
                % 2D image dataset (similar to grayscale)
                for idx = 1:size(data_per_block, 3)
                    im = data_per_block(:, :, idx);        % Assumption 2D image

                    % Input to the disk
                    % For regression problems
                    if (self.RegressionMode)
                        cls_lbl = [];
                    end

                    % Absolute filename to save data
                    subdir = fullfile(self.save_dirpath__, 'val', num2str(cls_lbl));
                    [~,~,~] = mkdir(subdir);
                    filename = fullfile(subdir, [num2str(self.global_val_idx__) '_' prefix '_' num2str(idx) '.tiff']);

                    % Write pp data (Matlab limitation: ValidationData: No support for datasource)
                    imsave_tiff(repmat(imresize(im, self.params_.TargetImageSize), 1, 1, 3), filename);

                    % Build the val_table (for classification problems)
                    if (~self.RegressionMode)
                        row = cell(1, 2);
                        row{1, 1} = filename;
                        row{1, 2} = categorical(cls_lbl);
                        val_table(self.global_val_idx__, :) = cell2table(row);
                        self.global_val_idx__ = self.global_val_idx__ + 1;
                        continue;
                    end
                    
                    tgt = target_per_block(:, idx)';
                    
                    % Build the val_table (in disk)
                    if (self.params_.FullDiskCompatibility)
                        % In this mode all data will be written to the disk
                        row = cell(1, 1);
                        row{1, 1} = filename;
                        val_table(self.global_val_idx__, :) = cell2table(row);

                    else
                        % Build the val_table (in memory)
                        row = cell(1, size(tgt, 2) + 1);
                        row{1, 1} = filename;
                        row(1, 2:end) = num2cell(tgt);   % Matlab limitation: ValidationData: No support for datasource
                        val_table(self.global_val_idx__, :) = cell2table(row);                    
                    end
                    
                    % Output to the disk (only for regression problems)
                    self.output_matobj__.ValOutput(self.global_val_idx__, :) = tgt;
                    self.global_val_idx__ = self.global_val_idx__ + 1;
                end
            end
            
            % Write testing split
            if (btype == MovingWindowPreProcessor.BLOCK_TE)
                % 2D image dataset (similar to grayscale)
                for idx = 1:size(data_per_block, 3)
                    im = data_per_block(:, :, idx);        % Assumption 2D image

                    % Input to the disk
                    % For regression problems
                    if (self.RegressionMode)
                        cls_lbl = [];
                    end

                    % Absolute filename to save data
                    subdir = fullfile(self.save_dirpath__, 'te', num2str(cls_lbl));
                    [~,~,~] = mkdir(subdir);
                    filename = fullfile(subdir, [num2str(self.global_te_idx__) '_' prefix '_' num2str(idx) '.tiff']);

                    % Write pp data (For ease of testing, resize the image and and write)
                    imsave_tiff(repmat(imresize(im, self.params_.TargetImageSize), 1, 1, 3), filename);

                    % Build the te_table                
                    if (~self.RegressionMode)
                        row = cell(1, 2);
                        row{1, 1} = filename;
                        row{1, 2} = categorical(cls_lbl);
                        te_table(self.global_te_idx__, :) = cell2table(row);
                        self.global_te_idx__ = self.global_te_idx__ + 1;
                        continue;

                    else
                        row = cell(1, 1);
                        row{1, 1} = filename;
                        te_table(self.global_te_idx__, :) = cell2table(row);                    
                    end

                    % Output to the disk (only for regression problems)
                    self.output_matobj__.TeOutput(self.global_te_idx__, :) = target_per_block(:, idx)';
                    self.global_te_idx__ = self.global_te_idx__ + 1;
                end
            end            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.IsInitialized(self)
            % ISINITIALIZED: Whether the `self` object is initialized.
            value = ~isempty(self.st__);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.HasFitPipelines(self)
            % HASFITPIPELINES: Whether the `self` object is initialized.
            value = ~isempty(self.fit_pipelines__);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.RegressionMode(self)
            % REGRESSIONMODE: Operating in regression mode or classification mode.
            % In regression mode: tables will be returned with targets.
            % In classification mode: pre-processed data will be written to class folders.
            value = (self.params_.OutputNodes > 0);
        end                
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

