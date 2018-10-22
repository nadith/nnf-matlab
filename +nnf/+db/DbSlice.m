classdef DbSlice
    % DBSLICE peforms slicing of nndb with the help of a selection structure.
    % IMPL_NOTES: Static class (thread-safe).
    %
    % Selection Structure (with defaults)
    % -----------------------------------
    % sel.tr_col_indices        = [];   % Training column indices
    % sel.tr_noise_rate         = [];   % Noise rate or Noise types for `tr_col_indices`
        
    % sel.tr_occlusion_rate     = [];   % Occlusion rate for `tr_col_indices`
    % sel.tr_noise_field        = [];   % Noise field for `tr_col_indices`
    % sel.tr_occlusion_field    = [];   % Occlusion field for `tr_col_indices`
    % sel.tr_illumination_field = [];   % Illumination field for `tr_col_indices`
    % sel.tr_out_col_indices    = [];   % Training target column indices
    % sel.val_col_indices       = [];   % Validation column indices
    % sel.val_out_col_indices   = [];   % Validation target column indices
    % sel.te_col_indices        = [];   % Testing column indices
    % sel.te_out_col_indices    = [];   % Testing target column indices
    % sel.nnpatches             = [];   % NNPatch object array
    % sel.use_rgb               = true; % Use rgb or convert to grayscale
    % sel.color_index           = [];   % Specific color indices (set .use_rgb = false)             
    % sel.use_real              = false;% Use real valued database TODO: (if .normalize = true, Operations ends in real values)
    % sel.scale                 = [];   % Scaling factor (resize factor)
    % sel.normalize             = false;% Normalize (0 mean, std = 1)
    % sel.histeq                = false;% Histogram equalization
    % sel.histmatch_col_index   = []    % Histogram match reference column index
    % sel.class_range           = [];   % Class range for training database or all (tr, val, te)
    % sel.val_class_range       = [];   % Class range for validation database
    % sel.te_class_range        = [];   % Class range for testing database
    % sel.pre_process_script    = [];   % Custom preprocessing script
    %
    % i.e
    % Pass a selection structure to split the nndb accordingly
    % [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs_tr, nndbs_val, nndbs_te, nndbs_tr_out, nndbs_val_out, nndbs_te_out, edatasets] = ...
                                                                    slice(nndb, sel, data_generator, pp_param, savepath) 
            % SLICE: slices the database according to the selection structure.
            % IMPL_NOTES: The new db will contain img at consecutive locations for duplicate indices
            % i.e Tr:[1 2 3 1], DB:[1 1 2 3]
            %
            % Parameters
            % ----------
        	% nndb : NNdb
            %     NNdb object that represents the dataset.
            %
            % sel : selection structure
            %     Information to split the dataset. (Ref: class documentation)            
            %
            % Returns
            % -------
            % nndbs_tr : NNdb or vector- NNdb
            %      Training NNdb object(s). (Incase patch division is required).
            %
            % nndbs_val : NNdb or vector- NNdb
            %      Validation NNdb object(s). (Incase patch division is required).
            %
            % nndbs_te : NNdb or vector- NNdb
            %      Testing NNdb object(s). (Incase patch division is required).   
            %
            % nndbs_tr_out : NNdb or vector- NNdb
            %      Training target NNdb object(s). (Incase patch division is required).
            %
            % nndbs_val_out : NNdb or vector- NNdb
            %      Validation target NNdb object(s). (Incase patch division is required).
            %
            % nndbs_te_out : NNdb or vector- NNdb
            %      Testing target NNdb object(s). (Incase patch division is required).
            %
            % vector : Dataset -vector
            %      List of dataset enums. Dataset order returned above.
            %

            % Imports
            import nnf.db.DbSlice; 
            import nnf.db.Format;
            import nnf.pp.im_pre_process;
            import nnf.core.iters.memory.DskmanMemDataIterator;
            import nnf.db.Dataset;
            import nnf.db.NNdb;
            import nnf.utl.Map;

            % Set defaults for arguments
            if (isempty(sel))
                sel = Selection();
                unq_n_per_cls = unique(nndb.n_per_class);
                
                if (isscalar(unq_n_per_cls))
                    sel.tr_col_indices = np.arange(unique(nndb.n_per_class));
                else
                    error(['Selection is not provided.'
                            ' tr_col_indices = nndb.n_per_class'
                            ' must be same for all classes']);
                end
            end           
            
          	% Error handling for arguments
            if (isempty(sel.tr_col_indices) && ...
                isempty(sel.tr_out_col_indices) && ...
                isempty(sel.val_col_indices) && ...
                isempty(sel.val_out_col_indices) && ...
                isempty(sel.te_col_indices) && ...
                isempty(sel.te_out_col_indices))
                error('ARG_ERR: [tr|tr_out|val|val_out|te]_col_indices: mandatory field');
            end            
            if ((~isempty(sel.use_rgb) && sel.use_rgb) && ~isempty(sel.color_indices))
                error('ARG_CONFLICT: sel.use_rgb, sel.color_indices');
            end

            % Set defaults
            if (nargin < 5)
                savepath = [];
            end
            
            % Set defaults for data generator   
            if (nargin < 3 || isempty(data_generator))
                if (nargin < 4); pp_param = []; end
                data_generator = DskmanMemDataIterator(pp_param);
            end
            data_generator.init_params(nndb);
            
            % Set default for sel.use_rgb
            if (isempty(sel.use_rgb))
                sel.use_rgb = true;
                disp('[DEFAULT] selection.use_rgb (None): True');
            end

            % Set defaults for class range (tr|val|te|...)
            cls_range = sel.class_range;
            if (isempty(cls_range)); sel.class_range = 1:nndb.cls_n; end   

            % Initialize class ranges and column ranges
            % DEPENDENCY -The order must be preserved as per the enum Dataset 
            % REF_ORDER: [TR=1, VAL=2, TE=3, TR_OUT=4, VAL_OUT=5]
            cls_ranges = {sel.class_range ... 
                            sel.val_class_range ...
                            sel.te_class_range ...
                            sel.class_range ... 
                            sel.val_class_range ...
                            sel.te_class_range};

            col_ranges = {sel.tr_col_indices ... 
                        sel.val_col_indices ...
                        sel.te_col_indices ...
                        sel.tr_out_col_indices ... 
                        sel.val_out_col_indices ...
                        sel.te_out_col_indices};

            % Set defaults for class ranges [val|te|tr_out|val_out]
            [cls_ranges, col_ranges] = DbSlice.set_default_cls_range_(1, cls_ranges, col_ranges);
            
            % NOTE: TODO: Whitening the root db did not perform well (co-variance is indeed needed)

            % Initialize NNdb dictionary
            % i.e dict_nndbs(Dataset.TR) => {NNdb_Patch_1, NNdb_Patch_2, ...}
            dict_nndbs = DbSlice.init_dict_nndbs_(col_ranges);

            % Set patch iterating loop's max count and nnpatch list
            patch_loop_max_n = 1;
            nnpatch_list = [];

            if (~isempty(sel.nnpatches))
                % Must have atleast 1 patch to represent the whole image
                nnpatch = sel.nnpatches(1);
            
                % PERF: If there is only 1 patch and it is the whole image
                if ((length(sel.nnpatches) == 1) && nnpatch.is_holistic)
                    % Pass

                else
                    nnpatch_list = sel.nnpatches;
                    patch_loop_max_n = length(sel.nnpatches);
                end
            end

            % Initialize the generator
            data_generator.init_ex(cls_ranges, col_ranges, true);       

            % LIMITATION: PYTHON-MATLAB (no support for generators)
            % [PERF] Iterate through the chosen subset of the nndb database
            [cimg, ~, cls_idx, col_idx, datasets, stop] = data_generator.next();
            while (~stop)

                % Perform pre-processing before patch division
                % Perform image operations only if db_format comply them
                if (nndb.db_format == Format.H_W_CH_N || nndb.db_format == Format.N_H_W_CH)

                    % For histogram equalizaion operation (cannonical image)
                    cann_cimg = [];
                    if (~isempty(sel.histmatch_col_index))
                        [cann_cimg, ~] = data_generator.get_cimg_frecord(cls_idx, sel.histmatch_col_index);
                        % Alternative:
                        % cls_st = nndb.cls_st(sel.histmatch_col_index)
                        % cann_cimg = nndb.get_data_at(cls_st)
                    end

                    % Perform image preocessing
                    cimg = DbSlice.preprocess_im_with_sel_(cimg, cann_cimg, sel, data_generator.get_im_ch_axis());
                end
                
                % Iterate through image patches
                for pi=1:patch_loop_max_n
                        
                    % Holistic image (by default)
                    pimg = cimg;

                    % Generate the image patch
                    if (~isempty(nnpatch_list))                        
                        nnpatch = nnpatch_list(pi);
                        x = nnpatch.offset(2);
                        y = nnpatch.offset(1);                                                                                
                        w = nnpatch.w;
                        h = nnpatch.h;
                        
                        % Target dbs for input patches might be holistic
                        if (nnpatch.is_holistic)
                            % pass
                        % Extract the patch
                        elseif (nndb.db_format == Format.H_W_CH_N || nndb.db_format == Format.N_H_W_CH)
                            pimg = cimg(y:y+h-1, x:x+w-1, :);
                        
                        elseif (nndb.db_format == Format.H_N || nndb.db_format == Format.N_H)
                            % 1D axis is `h` if w > 1, otherwise 'w'
                             if (w > 1); pimg = cimg(x:x+w-1); else; pimg = cimg(y:y+h-1); end
                        end
                    end

                    % The offset/index for the col_index in the tr_col_indices vector
                    tci_offsets = [];
                    
                    % Iterate through datasets
                    for dsi=1:numel(datasets)
                        dataset = datasets{dsi};
                        edataset = dataset{1};
                        is_new_class = dataset{2};
                                              
                        % Fetch patch databases, if None, create
                        nndbs = dict_nndbs(uint32(edataset));
                        if (isempty(nndbs))
                            % Add an empty NNdb for all `nnpatch` on first edataset entry
                            for pi_tmp=1:patch_loop_max_n
                                nndbs(end+1) = NNdb([Dataset.str(edataset) '_p' num2str(pi_tmp)], [], [], false, [], nndb.db_format);
                            end

                            % Update the dict_nndbs
                            dict_nndbs(uint32(edataset)) = nndbs;
                        end    

                        % Build Training DB
                        if (edataset == Dataset.TR)
                            
                            tci_offset = [];
                            
                            % If not set already
                            if isempty(tci_offsets)                            
                                % If noise or occlusion or illumination is required
                                if (~isempty(sel.tr_noise_field) || ...
                                    ~isempty(sel.tr_occlusion_field) || ...
                                    ~isempty(sel.tr_illumination_field))
                                    tci_offsets = find(sel.tr_col_indices == col_idx);
                                    tci_offset = tci_offsets(dsi);
                                end                                
                            else
                                tci_offset = tci_offsets(dsi);
                            end
                            
                            % PERF: First use of user_data
                            if isempty(data_generator.user_data)
                                data_generator.user_data = Map();
                            end
                            
                            user_data = data_generator.user_data.setdefault(Dataset.str(edataset), []);
                            [~, user_data] = DbSlice.build_nndb_tr_(nndbs, pi, is_new_class, pimg, tci_offset, ...
                                                            sel.tr_noise_field, ...
                                                            sel.tr_occlusion_field, ...
                                                            sel.tr_illumination_field, ...
                                                            user_data);
                                                        
                            data_generator.user_data(Dataset.str(edataset)) = user_data;
                            
                        % Build Training Target DB
                        elseif (edataset == Dataset.TR_OUT)
                            DbSlice.build_nndb_tr_out_(nndbs, pi, is_new_class, pimg);

                        % Build Valdiation DB
                        elseif (edataset == Dataset.VAL)
                            DbSlice.build_nndb_val_(nndbs, pi, is_new_class, pimg);

                        % Build Valdiation Target DB    
                        elseif (edataset == Dataset.VAL_OUT)
                            DbSlice.build_nndb_val_out_(nndbs, pi, is_new_class, pimg);

                        % Build Testing DB
                        elseif (edataset == Dataset.TE)
                            DbSlice.build_nndb_te_(nndbs, pi, is_new_class, pimg);

                        % Build Testing Target DB
                        elseif (edataset == Dataset.TE_OUT)
                            DbSlice.build_nndb_te_out_(nndbs, pi, is_new_class, pimg);
                            
                        end
                    end
                end
                [cimg, ~, cls_idx, col_idx, datasets, stop] = data_generator.next();
            end

            % Returns NNdb object instead of NNdb array (non patch requirement)
            if (isempty(sel.nnpatches)) 
                
                % Save the splits in the disk
                if ~isempty(savepath)        
                    [filepath, name, ~] = fileparts(savepath);
                    datasets = Dataset.get_enum_list();
                    for i=1:numel(datasets)
                        dataset = datasets(i);                        
                        tmp_nndb = DbSlice.p0_nndbs(dict_nndbs, uint32(dataset))
                        if ~isempty(tmp_nndb)                            
                            tmp_nndb.save(fullfile(filepath, name) + "_" + str.upper(str(dataset)) + ".mat");
                        end
                    end
                end
                
                nndbs_tr = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TR));
                nndbs_val = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.VAL));
                nndbs_te = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TE));
                nndbs_tr_out = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TR_OUT));
                nndbs_val_out = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.VAL_OUT));
                nndbs_te_out = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TE_OUT));
                edatasets = [Dataset.TR Dataset.VAL Dataset.TE Dataset.TR_OUT Dataset.VAL_OUT Dataset.TE_OUT];
                return;
            end  
                        
            % Save the splits in the disk
            if ~isempty(savepath)
                [filepath, name, ~] = fileparts(savepath);
                datasets = Dataset.get_enum_list();
                for i=1:numel(datasets)
                    dataset = datasets(i); 
                    tmp_nndbs = [];
                    if ~isempty(dict_nndbs(uint32(dataset))); tmp_nndbs = dict_nndbs(uint32(dataset)); end                        
                    if ~isempty(tmp_nndbs)
                        nndb_count = numel(tmp_nndbs);
                        for pi=1:nndb_count
                            tmp_nndb = tmp_nndb(pi);
                            tmp_nndb.save(fullfile(filepath, name) + "_" + str.upper(str(dataset)) + "_" + str(pi) + ".mat");
                        end
                    end                    
                end
            end
            
            nndbs_tr = dict_nndbs(uint32(Dataset.TR));
            nndbs_val = dict_nndbs(uint32(Dataset.VAL));
            nndbs_te = dict_nndbs(uint32(Dataset.TE));
            nndbs_tr_out = dict_nndbs(uint32(Dataset.TR_OUT));
            nndbs_val_out = dict_nndbs(uint32(Dataset.VAL_OUT));
            nndbs_te_out = dict_nndbs(uint32(Dataset.TE_OUT));
            edatasets = [Dataset.TR Dataset.VAL Dataset.TE Dataset.TR_OUT Dataset.VAL_OUT Dataset.TE_OUT];
        end    
   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function cimg = preprocess_im_with_sel_(cimg, cann_cimg, sel, ch_axis)
            % Perform image preprocessing with compliance to selection object.
            % 
            % Parameters
            % ----------
            % cimg : ndarray -uint8
            %     3D Data tensor to represent the color image.
            % 
            % cann_cimg : ndarray
            %     Cannonical/Target image (corresponds to sel.tr_out_indices).
            % 
            % sel : :obj:`Selection`
            %     Information to pre-process the dataset. (Ref: class documentation).
            % 
            % ch_axis : int
            %     Color axis of the image.
            % 
            % Returns
            % -------
            % ndarray -uint8
            %     Pre-processed image.
            %
            
            % Imports
            import nnf.db.DbSlice;
            import nnf.pp.im_pre_process;
        
            % Image resize
            if (~isempty(sel.scale))
                cimg = imresize(cimg, sel.scale);
                if (~isempty(cann_cimg))
                    cann_cimg = imresize(cann_cimg, sel.scale);
                end
            end

            % Color / Gray Scale Conversion (if required)
            cimg = DbSlice.process_color_(cimg, sel);
            if (~isempty(cann_cimg))
                cann_cimg = DbSlice.process_color_(cann_cimg, sel);
            end

            % Image pre-processing parameters                        
            pp_params.histeq    = sel.histeq;
            pp_params.normalize = sel.normalize;
            pp_params.histmatch = ~isempty(sel.histmatch_col_index);
            pp_params.cann_img  = cann_cimg;
            pp_params.ch_axis   = ch_axis;
            cimg = im_pre_process(cimg, pp_params);

            % [CALLBACK] the specific pre-processing script
            if (~isempty(sel.pre_process_script))
                cimg = sel.pre_process_script(cimg);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function examples(imdb, im_per_class)
            % Extensive set of examples.
            % 
            % Parameters
            % ----------
            % imdb : 4D tensor -uint8
            %     NNdb object that represents the dataset. Assume it contain only 8 images per subject.
            % 
            %     Format: (Samples x H x W x CH).
            % 
            % im_per_class : int
            %     Image per class.

            
            % Imports
            import nnf.db.NNdb;
            import nnf.db.DbSlice;
            import nnf.db.Selection;
            import nnf.core.generators.NNPatchGenerator;            
            import nnf.db.sel.NoiseField
            import nnf.db.sel.OcclusionField
            import nnf.db.sel.IlluminationField
                        
            % 
            % Select 1st 2nd 4th images of each identity for training
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4]; %[1 2 4]; 
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel); % nndb_tr = DbSlice.slice(nndb, sel); 

            
            %
            % Select 1st 2nd 4th images of each identity for training.
            % Divide into patches
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4]; %[1 2 4];             
            patch_gen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33);
            sel.nnpatches = patch_gen.generate_nnpatches();

            % vector of NNdb objects for each patch
            [nndbs_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            nndbs_tr(1).show(10, 3)
            figure, 
            nndbs_tr(2).show(10, 3)
            figure, 
            nndbs_tr(3).show(10, 3)
            figure,
            nndbs_tr(4).show(10, 3) 
            
            
            % 
            % Select 1st 2nd 4th images of each identity for training
            % Select 3rd 5th images of each identity for testing
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
           	% 
            % Select 1st 2nd 4th images of identities denoted by class_range for training
            % Select 3rd 5th images of identities denoted by class_range for testing   
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            %
            % Select 1st and 2nd image from 1st class and 2nd, 3rd and 5th image from 2nd class for training 
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices      = {[1 2], [2 3 5]};
            sel.class_range         = [1:2];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training
            % Select 1st 2nd 4th images images of identities denoted by class_range for validation  
            % Select 3rd 5th images of identities denoted by class_range for testing
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.val_col_indices= [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            [nndb_tr, nndb_val, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training
            % Select 1st 2nd 4th images images of identities denoted by val_class_range for validation   
            % Select 3rd 5th images of identities denoted by te_class_range for testing
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.val_col_indices= [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            sel.val_class_range= [6:15];
            sel.te_class_range = [17:20];
            [nndb_tr, nndb_val, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training
            % Select 3rd 4th images of identities denoted by val_class_range for validation
            % Select 3rd 5th images of identities denoted by te_class_range for testing
            % Select 1st 1st 1st images of identities denoted by class_range for training target
            % Select 1st 1st images of identities denoted by val_class_range for validation target
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices      = [1:2 4];
            sel.val_col_indices     = [3 4];
            sel.te_col_indices      = [3 5]; 
            sel.tr_out_col_indices  = [1 1 1];
            sel.val_out_col_indices = [1 1];
            sel.te_out_col_indices  = [1 1]; 
            sel.class_range         = [1:10];
            sel.val_class_range     = [6:15];
            sel.te_class_range      = [17:20];
            [nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out, nndb_te_out, ~] = DbSlice.slice(nndb, sel);
            nndb_tr.show(10, 3)
            figure, nndb_val.show(10, 2)
            figure, nndb_te.show(4, 2)
            figure, nndb_tr_out.show(10, 3)            
            figure, nndb_val_out.show(10, 2)
            figure, nndb_te_out.show(10, 2)


            %
            % Using special enumeration values
            % Training column will consists of first 60% of total columns available
            % Testing column will consists of first 40% of total columns available
            nndb = NNdb('original', imdb, im_per_class, True);
            sel = Selection();
            sel.tr_col_indices      = Select.PERCENT_60;
            sel.te_col_indices      = Select.PERCENT_40;
            sel.class_range         = [1:10];
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            nndb_tr.show(10, 5);
            nndb_te.show(10, 4);


            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add various noise types @ random locations of varying degree.
            %               default noise type: random black and white dots.
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4];            
            % Param1: Noise type
            % Param2: Percentage of noise
            sel.tr_noise_field = NoiseField(['c' 'c' 'c'], [0 0.5 0.2]);
            % sel.tr_noise_field = NoiseField(['g' 'g' 'g'], []);
            sel.class_range    = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);

            
            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add various occlusion types ('t':top, 'b':bottom, 'l':left, 'r':right, 'h':horizontal, 'v':vertical) of varying degree.
            %               default occlusion type: 'b'.
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];            
            % Param1: Occlusion type: 't' for selected tr. indices [1, 2, 4] 
            % Param2: Percentage of occlusion
            sel.tr_occlusion_field = OcclusionField('ttt', [0 0.5 0.2]);
            %sel.tr_occlusion_field = OcclusionField('tbr', [0 0.5 0.2]);
            %sel.tr_occlusion_field = OcclusionField('lrb', [0 0.5 0.2]);            
            sel.class_range = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add occlusions in the middle (horizontal/vertical).
            %               default occlusion type: 'b'.
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];
            % Param1: Occlusion type: 't' for selected tr. indices [1, 2, 4] 
            % Param2: Percentage of occlusion
            % Param3: Offset from top since occlusion type = t
            sel.tr_occlusion_field = OcclusionField('ttt', [0 0.5 0.2], [0 0.25 0.4]);
            sel.class_range = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);

            
            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               use custom occlusion filter for 66x66 images.
            occl_filter = ones(66, 66);
            for i=1:33
                for j=1:34-i
                    occl_filter(i, j) = 0;
                end
            end
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];
            % Param4: Custom occlusion filter
            sel.tr_occlusion_field = OcclusionField([], [], [], occl_filter);
            sel.class_range = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add illumination, place the light source @ [0 33] position.
            sel = Selection();
            sel.tr_col_indices = [1:2 4];
            % Param1: Illumination position / Light source placement 
            % Param2: Covariance
            % Param3: Brightness factor (0-1)
            % Param4: Darkness factor (0-1)
            % Param5: Thresholds (to eliminate the small noise values generated)
            %           Capture when a pixel value is shifted by 1 due to the light source
            sel.tr_illumination_field  = IlluminationField([0 33], {[1 0; 0 1], [1 0; 0 1], [10 0; 0 10]}, 0);
            sel.class_range = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
           	% 
            % To prepare regression datasets, training dataset and training target dataset
            % Select 1st 2nd 4th images of each identity for training.
            % Select 1st 1st 1st image of each identity for corresponding training target.
            % Select 3rd 5th images of each identity for testing.
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];       %[1 2 4]; 
            sel.tr_out_col_indices = [1 1 1];   %[1 1 1]; (Regression 1->1, 2->1, 4->1) 
            sel.te_col_indices = [3 5];         %[3 5]; 
            [nndb_tr, ~, nndb_te, nndb_tr_out, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %  
            % Resize images by 0.5 scale factor.
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.scale          = 0.5;
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %  
            % Use gray scale images.
            % Perform histogram equalization.
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.histeq         = true;
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %  
            % Use gray scale images.
            % Perform histogram match. This will be performed with the 1st image of each identity
            % irrespective of the selection choice. (refer code for more details)
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.histmatch_col_index = 1;
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %
            % If imdb_8 supports many color channels
            nndb = NNdb('original', imdb, im_per_class, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.color_indices  = 5;         % color channel denoted by 5th index
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public, Static) % ?nnf.alg.PCA
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [oc_filter] = get_occlusion_patch(h, w, dtype, occl_type, occl_rate, occl_offset)
            % Get a occlusion patch to place on top of an image.
            % 
            % Parameters
            % ----------
            % h : int
            %     Height of the occlusion patch.
            % 
            % h : int
            %     Width of the occlusion patch.
            % 
            % dtype : str or dtype
            %     Typecode or data-type to which the array is cast.
            % 
            % occl_type : char
            %     Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right).
            % 
            % occl_rate : double
            %     Occlusion ratio.
            % 
            % occl_offset : double
            %     Occlusion start offset (as a ratio) from top/bottom/left/right corner depending on `occl_type`.
            % 
            % Returns
            % -------
            % img : 2D tensor -uint8
            %      Occlusion filter. (ones and zeros).
            %
            
            % Set defaults for arguments
            if (nargin < 6 || isempty(occl_offset)); occl_offset = 0; end
            
            oc_filter = ones(h, w);
            oc_filter = cast(oc_filter, dtype);
            
            if (isempty(occl_type) || occl_type == 'b')
                sh = ceil(occl_rate * h);
                en = floor((1-occl_offset) * h); 
                st = en - sh + 1; if (st < 0); st = 1; end
                oc_filter(st:en, 1:w) = 0;

            elseif (occl_type == 'r')
                sh = ceil(occl_rate * w);
                en = floor((1-occl_offset) * w); 
                st = en - sh + 1; if (st < 0); st = 1; end
                oc_filter(1:h, st:en) = 0;

            elseif (occl_type == 't' || occl_type == 'v')
                sh = floor(occl_rate * h);
                st = floor(occl_offset * h) + 1; 
                en = st + sh - 1; if (en > h); en = h; end
                oc_filter(st:en, 1:w) = 0;

            elseif (occl_type == 'l' || occl_type == 'h')
                sh = floor(occl_rate * w);
                st = floor(occl_offset * w) + 1; 
                en = st + sh - 1; if (en > w); en = w; end
                oc_filter(1:h, st:en) = 0;              
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end    
    
	methods (Access = protected, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndbs = p0_nndbs_(dict_nndbs, ekey)
            if (~isempty(dict_nndbs(ekey)))
                nndbs = dict_nndbs(ekey);
                nndbs = nndbs(1);
            else
                nndbs = [];
            end
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function  [cls_ranges, col_ranges] = set_default_cls_range_(default_idx,  cls_ranges, col_ranges)
            % SET_DEFAULT_RANGE: Sets the value of the class range at default_idx 
            % to the class ranges at other indices [val|te|tr_out|val_out]
            %
            % Parameters
            % ----------
            % default_idx : int
            %     Index for list of class ranges. Corresponds to the enumeration `Dataset`.
            % 
            % cls_ranges : list of :obj:`list`
            %     Class range for each dataset. Indexed by enumeration `Dataset`.
            % 
            % col_ranges : list of :obj:`list`
            %     Column range for each dataset. Indexed by enumeration `Dataset`.
            % 
            for rng_idx=1:numel(col_ranges)
                if (~isempty(col_ranges{rng_idx}) && ...
                    isempty(cls_ranges{rng_idx}))
                    cls_ranges{rng_idx} = cls_ranges{default_idx};
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function dict_nndbs = init_dict_nndbs_(col_ranges)
            % DICT_NNDBS: Initializes a dictionary that tracks the `nndbs` for each dataset.
            % 
            %     i.e dict_nndbs[Dataset.TR] => [NNdb_Patch_1, NNdb_Patch_2, ...]
            % 
            % Parameters
            % ----------
            % col_ranges : list of :obj:`list`
            %       Column range for each dataset. Indexed by enumeration `Dataset`.            
            % 
            % Returns
            % -------
            % dict_nndbs : Map of :obj:`NNdb`
            %       Dictionary of nndbs. Keyed by enumeration `Dataset`.

            % Import
            import nnf.db.Dataset;
            import nnf.db.DbSlice;
            import nnf.db.NNdb;

            % LIMITATION: Does not support enum key types
            dict_nndbs = containers.Map(uint32(Dataset.TR), uint16(zeros(2)));
            remove(dict_nndbs, uint32(Dataset.TR));

            % Iterate through ranges
            for ri=1:numel(col_ranges)
                dict_nndbs(ri) = NNdb.empty;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [img] = process_color_(img, sel) 
            % PROCESS_COLOR: performs color related functions.
            % 
            % Parameters
            % ----------
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % sel : :obj:`Selection`
            %     Selection object with the color processing fields.
            %
            % Returns
            % -------
            % img : 3D tensor -uint8
            %      Color processed image.
            %
            if (isempty(img)); return; end        
            [~, ~, ch] = size(img);            

            % Color / Gray Scale Conversion (if required) 
            if (~(sel.use_rgb))

                % if image has more than 3 channels
                if (ch >= 3)                                    
                    if (numel(sel.color_indices) > 0)
                        X = img(:, :, sel.color_indices);                       
                    else
                        X = rgb2gray(img);
                    end

                else
                    X = img(:, :, 1);
                end

                img = X;  
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, user_data] = build_nndb_tr_(nndbs, pi, is_new_class, img, ...
                                            index, noise_field, occl_field, illu_field, user_data) 
            % BUILD_NNDB_TR: builds the nndb training database.
            %          
            % Parameters
            % ----------
            % nndbs : :obj:`List` of :obj:`NNdb`
            %     List of nndbs each corresponds to patches.
            % 
            % pi : int
            %     Patch index that used to index `nndbs`.
            % 
            % is_new_class : int
            %     Whether it's a new class or not.
            % 
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % noise_rate : double
            %     Noise ratio.
            % 
            % occl_rate : double
            %     Occlusion ratio.
            % 
            % occl_type : char
            %     Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right).
            % 
            % occl_offset : double
            %     Occlusion start offset (as a ratio) from top/bottom/left/right corner depending on `occl_type`.
            % 
            % Returns
            % -------
            % nndbs : Array of :obj:`NNdb`
            %     NNdb objects for each patch.
            %  
            
            % Imports 
            import nnf.db.DbSlice;
            import nnf.db.Format;
             
            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
                
            % Add different noise depending on the type or rate
            if ~isempty(noise_field); img = DbSlice.add_noise_(img, noise_field, index); end
            
            % Adding user-defined occlusion filter or different occlusions depending on the percentage
            if ~isempty(occl_field); [img, user_data] = DbSlice.add_occlusion_(img, occl_field, index, user_data); end
            
            % Adding illumination light source
            if ~isempty(illu_field); [img, user_data] = DbSlice.add_illumination_(img, illu_field, index, user_data); end                
            
            % Add data to nndb
            nndb.add_data(img);

            % Update the properties of nndb
            nndb.update_attr(is_new_class);        
        end
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndbs = build_nndb_tr_out_(nndbs, pi, is_new_class, img)
            % BUILD_NNDB_TR_OUT: builds the nndb training target database.
            %            
            % Parameters
            % ----------
            % nndbs : :obj:`List` of :obj:`NNdb`
            %     List of nndbs each corresponds to patches.
            % 
            % pi : int
            %     Patch index that used to index `nndbs`.
            % 
            % is_new_class : int
            %     Whether it's a new class or not.
            % 
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % Returns
            % -------
            % nndbs : Array of :obj:`NNdb`
            %     NNdb objects for each patch.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
            nndb.add_data(img);
            
            % Update the properties of nndb
            nndb.update_attr(is_new_class);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndbs = build_nndb_val_(nndbs, pi, is_new_class, img)
            % BUILD_NNDB_VAL: builds the nndb validation database.
            %            
            % Parameters
            % ----------
            % nndbs : :obj:`List` of :obj:`NNdb`
            %     List of nndbs each corresponds to patches.
            % 
            % pi : int
            %     Patch index that used to index `nndbs`.
            % 
            % is_new_class : int
            %     Whether it's a new class or not.
            % 
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % Returns
            % -------
            % nndbs : Array of :obj:`NNdb`
            %     NNdb objects for each patch.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
            nndb.add_data(img);
            
            % Update the properties of nndb
            nndb.update_attr(is_new_class);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndbs = build_nndb_val_out_(nndbs, pi, is_new_class, img)
            % BUILD_NNDB_VAL_OUT: builds the nndb validation target database.
            %            
            % Parameters
            % ----------
            % nndbs : :obj:`List` of :obj:`NNdb`
            %     List of nndbs each corresponds to patches.
            % 
            % pi : int
            %     Patch index that used to index `nndbs`.
            % 
            % is_new_class : int
            %     Whether it's a new class or not.
            % 
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % Returns
            % -------
            % nndbs : Array of :obj:`NNdb`
            %     NNdb objects for each patch.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
            nndb.add_data(img);
            
            % Update the properties of nndb
            nndb.update_attr(is_new_class);
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndbs = build_nndb_te_(nndbs, pi, is_new_class, img)
            % BUILD_NNDB_TE: builds the testing database.
            %            
            % Parameters
            % ----------
            % nndbs : :obj:`List` of :obj:`NNdb`
            %     List of nndbs each corresponds to patches.
            % 
            % pi : int
            %     Patch index that used to index `nndbs`.
            % 
            % is_new_class : int
            %     Whether it's a new class or not.
            % 
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % Returns
            % -------
            % nndbs : Array of :obj:`NNdb`
            %     NNdb objects for each patch.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
            nndb.add_data(img);
            
            % Update the properties of nndb
            nndb.update_attr(is_new_class);
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndbs = build_nndb_te_out_(nndbs, pi, is_new_class, img)
            % BUILD_NNDB_TE: builds the testing database.
            %
            % Parameters
            % ----------
            % nndbs : :obj:`List` of :obj:`NNdb`
            %     List of nndbs each corresponds to patches.
            % 
            % pi : int
            %     Patch index that used to index `nndbs`.
            % 
            % is_new_class : int
            %     Whether it's a new class or not.
            % 
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % Returns
            % -------
            % nndbs : Array of :obj:`NNdb`
            %     NNdb objects for each patch.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
            nndb.add_data(img);
            
            % Update the properties of nndb
            nndb.update_attr(is_new_class);      
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function img = add_noise_(img, noise_field, index)
            % ADD_NOISE: Add noise to the image.
            % 
            % Parameters
            % ----------
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % noise_field : :obj:`NoiseField`
            %     Indicates the noise(s) for Selection.tr_col_indices.
            % 
            % index: int
            %     The index which the value is required at.
            % 
            % Returns
            % -------
            % ndarray -uint8
            %     Image with noise.
            %
        
            % Imports 
            import nnf.db.DbSlice;
            
            % Get noise information at `index`
            noise = noise_field.get_value_at(index);            
            if isempty(noise); return; end
            
            % Image dimensions
            [h, w, ch] = size(img);
            
            if (strcmp(noise.type, 'g'))
                img = imnoise(img, 'gaussian');
                % img = imnoise(img, 'gaussian');
                % img = imnoise(img, 'gaussian');
                % img = imnoise(img, 'gaussian');

            elseif (strcmp(noise.type, 'c'))
                % Perform random corruption
                % Corruption Size (H x W)
                cs = [uint16(h * noise.rate) uint16(w * noise.rate)];

                % Random location choice
                % Start of H, W (location)
                sh = 1 + rand()*(h-cs(1)-1);
                sw = 1 + rand()*(w-cs(2)-1);

                % Set the corruption
                cimg = uint8(DbSlice.rand_corrupt_(cs(1), cs(2)));

                if (ch == 1)
                    img(sh:sh+cs(1)-1, sw:sw+cs(2)-1) = cimg;
                else
                    for ich=1:ch
                        img(sh:sh+cs(1)-1, sw:sw+cs(2)-1, ich) = cimg;
                    end
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [img, user_data] = add_occlusion_(img, occlusion_field, index, user_data)
            % ADD_OCCLUSION: Add occlusion to the image.
            % 
            % Parameters
            % ----------
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % occlusion_field : :obj:`OcclusionField`
            %     Indicates the occlusion(s) for Selection.tr_col_indices.
            % 
            % index: int
            %     The index which the value is required at.
            % 
            % user_data : :obj:`dict`
            %     PERF: Dictionary to store reusable calculated content during occlusion addition.
            % 
            % Returns
            % -------
            % ndarray -uint8
            %     Image with occlusion.
            % 
            % user_data : :obj:`dict`
            %     Updated user_data content.
            %
            
            % Imports 
            import nnf.db.DbSlice;
            import nnf.utl.Map;
            
            % Get occlusion information at `index`
            occl = occlusion_field.get_value_at(index);            
            if isempty(occl); return; end

            % Image dimensions
            [h, w, ch] = size(img);
            
            % If a custom oc_filter is specified
            occl_filter = occl.filter;

            if isempty(occl_filter)
                
                % PERF: Load it if available
                if isfield(user_data, 'occl')
                    % ud -> occl -> type -> oc_rate -> oc_offset -> oc_filter
                    ud_oc_rates = user_data.occl.types.setdefault(occl.type, Map('double'));
                    ud_oc_offsets = ud_oc_rates.setdefault(occl.rate, Map('int32'));
                    occl_filter = ud_oc_offsets.setdefault(occl.offset, []);
                end
                
                if isempty(occl_filter)                
                    occl_filter = DbSlice.get_occlusion_patch(h, w, class(img), occl.type, occl.rate, occl.offset);
                    
                    % PERF: Save it for next iteration
                    % ud -> occl -> type -> oc_rate -> oc_offset -> oc_filter
                    if ~isfield(user_data, 'occl'); user_data.occl.types = Map(); end
                    
                    ud_oc_rates = user_data.occl.types.setdefault(occl.type, Map('double'));
                    ud_oc_offsets = ud_oc_rates.setdefault(occl.rate, Map('int32'));
                    ud_oc_offsets.setdefault(occl.offset, occl_filter);
                end
            end

            % For grey scale
            if (ch == 1)
                img = uint8(occl_filter).*img;
            else
                % For colored
                img = repmat(uint8(occl_filter), 1, 1, ch).*img;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [img, user_data] = add_illumination_(img, illumination_field, index, user_data)
            % ADD_ILLUMINATION: Add illumination to the image.
            % 
            % Parameters
            % ----------
            % img : ndarray -uint8
            %     3D Data tensor. (format = H x W x CH)
            % 
            % illumination_field : :obj:`OcclusionField`
            %     Indicates the illumination(s) for Selection.tr_col_indices.
            % 
            % index: int
            %     The index which the value is required at.
            % 
            % user_data : :obj:`dict`
            %     PERF: Dictionary to store reusable calculated content during illumination addition.
            % 
            % Returns
            % -------
            % ndarray -uint8
            %     Image with illumination.
            % 
            % user_data : :obj:`dict`
            %     Updated user_data content.
            %
            
            % Imports
            import nnf.utl.Map;
            
            % Get illumination information at `index`
            illu = illumination_field.get_value_at(index);            
            if isempty(illu); return; end
            
            % PERF: Load it if available
            info = [];
            if isfield(user_data, 'illu')         
                % Only when positions and covariances are scalars
                if  ~iscell(illumination_field.positions) && ~iscell(illumination_field.covariances)
                    ud_il_darkness = user_data.illu.brightness.setdefault(illu.brightness, Map('double'));
                    ud_il_thresholds = ud_il_darkness.setdefault(illu.darkness, Map('double'));
                    info = ud_il_thresholds.setdefault(illu.threshold, []);
                end
            end
                
                            
            % Image dimensions
            [h, w, ch] = size(img);
                
            if ~isempty(info)
                illu_kernel = info.illu_kernel; 
                illu_filter_dark = info.illu_filter_dark;
                mask = info.mask;
                
            else               
                % Generate a 2-D gaussian noise matrix
                % Sparse Mesh
                x_range = (0 - illu.position(2))+1 : (h - illu.position(2));
                y_range = (0 - illu.position(1))+1 : (w - illu.position(1));
                
                % Initialize the noise kernal
                illu_kernel = zeros(numel(x_range), numel(y_range));
            
                for i=1:numel(x_range)
                    for j=1:numel(y_range)
                        vect = [x_range(i) y_range(j)];
                        illu_kernel(i, j) = exp(- (1/2) * (vect / illu.covariance) * vect');
                    end
                end
                            
                % Refining the mask (eliminate the small noise values generated)
                % Mask is used to average the pixels of the image
                mask = illu_kernel;
                mask(mask < illu.threshold) = 0;
                mask(mask >= illu.threshold) = 1;

                % For darker Areas
                illu_filter_dark = zeros(h, w);
                illu_filter_dark (~mask) = -255 * illu.darkness; % reduce by this factor
                                
                % PERF: Save it for next iteration
                % Only when positions and covariances are scalars
                if  ~iscell(illumination_field.positions) && ~iscell(illumination_field.covariances)
                    
                    info.illu_kernel = illu_kernel; 
                    info.illu_filter_dark = illu_filter_dark;
                    info.mask = mask;
                    
                    % ud -> illu -> brightness -> il_darkness -> il_thresholds
                    if ~isfield(user_data, 'illu')
                        user_data.illu.brightness = Map('double'); 
                    end
                    
                    ud_il_darkness = user_data.illu.brightness.setdefault(illu.brightness, Map('double'));
                    ud_il_thresholds = ud_il_darkness.setdefault(illu.darkness, Map('double'));
                    ud_il_thresholds.setdefault(illu.threshold, info);                    
                end
            end  
                        
            % For brighter Areas
            filtered = double(img) .* mask;            
            if (ch == 1)
                illu_scale = 255 - mean(filtered(filtered~=0)); % enhance by this difference                
            else
                % For each channel
                illu_scale = zeros(size(img));                
                for i=1:ch
                    tmp = filtered(:, :, i);
                    illu_scale(:, :, i) = 255 - mean(tmp(tmp~=0)); % enhance by this difference
                end
            end  
            
            img = uint8(double(img) + illu.brightness * illu_scale .* illu_kernel + illu_filter_dark);
        end    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function img = rand_corrupt_(height, width)
            % RAND_CORRUPT: corrupts the image with a (height, width) block.
            %
            % Parameters
            % ----------
            % height : int
            %     Height of the corruption block.
            %
            % width : int
            %     Width of the corruption block.
            %
            % Returns
            % -------
            % img : 3D tensor -uint8
            %     3D-Data tensor that contains the corrupted image.
            %
            
            percentageWhite = 50; % Alter this value as desired
            
            dotPattern = zeros(height, width);
            
            % Set the desired percentage of the elements in dotPattern to 1
            dotPattern(1:round(0.01*percentageWhite*numel(dotPattern))) = 1;
            
            % Seed the random number generator
            % rand('twister',100*sum(clock));
            
            % Randomly permute the element order
            dotPattern = reshape(dotPattern(randperm(numel(dotPattern))), height, width);
            img = dotPattern .* 255;
            
            % imagesc(dotPattern);
            % colormap('gray');
            % axis equal;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end