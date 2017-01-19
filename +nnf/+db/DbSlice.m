classdef DbSlice 
    % DBSLICE peforms slicing of nndb with the help of a selection structure.
    % IMPL_NOTES: Static class (thread-safe).
    %
    % Selection Structure (with defaults)
    % -----------------------------------
    % sel.tr_col_indices        = [];   % Training column indices
    % sel.tr_noise_mask         = [];   % Noisy tr. col indices (bit mask)
    % sel.tr_noise_rate         = [];   % Rate or noise types for the above field
    % sel.tr_out_col_indices    = [];   % Training target column indices
    % sel.val_col_indices       = [];   % Validation column indices
    % sel.val_out_col_indices   = [];   % Validation target column indices
    % sel.te_col_indices        = [];   % Testing column indices
    % sel.nnpatches             = [];   % NNPatch object array
    % sel.use_rgb               = true; % Use rgb or convert to grayscale
    % sel.color_index           = [];   % Specific color indices (set .use_rgb = false)             
    % sel.use_real              = false;% Use real valued database TODO: (if .normalize = true, Operations ends in real values)
    % sel.scale                 = [];   % Scaling factor (resize factor)
    % sel.normalize             = false;% Normalize (0 mean, std = 1)
    % sel.histeq                = false;% Histogram equalization
    % sel.histmatch             = false;% Histogram match (ref. image: first image of the class)
    % sel.class_range           = [];   % Class range for training database or all (tr, val, te)
    % sel.val_class_range       = [];   % Class range for validation database
    % sel.te_class_range        = [];   % Class range for testing database
    % sel.pre_process_script    = [];   % Custom preprocessing script
    %
    % i.e
    % Pass a selection structure to split the nndb accordingly
    % [nndb_tr, ~, ~, ~] = DbSlice.slice(nndb, sel);
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
        
    methods (Access = public, Static) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs_tr, nndbs_val, nndbs_te, nndbs_tr_out, nndbs_val_out] = slice(nndb, sel) 
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
            %     Information to split the dataset.
            % 
            %     i.e
            %       Selection Structure (with defaults)
            %       -----------------------------------
            %       sel.tr_col_indices      = [];   % Training column indices
            %       sel.tr_noise_mask       = [];   % Noisy tr. col indices (bit mask)
            %       sel.tr_noise_rate       = [];   % Rate or noise types for the above field
            %       sel.tr_out_col_indices  = [];   % Training target column indices
            %       sel.val_col_indices     = [];     % Validation column indices
            %       sel.val_out_col_indices = [];   % Validation target column indices
            %       sel.te_col_indices      = [];   % Testing column indices
            %       sel.nnpatches           = [];   % NNPatch object array
            %       sel.use_rgb             = true; % Use rgb or convert to grayscale
            %       sel.color_index         = [];   % Specific color indices (set .use_rgb = false)             
            %       sel.use_real            = false;% Use real valued database TODO: (if .normalize = true, Operations ends in real values)
            %       sel.scale               = [];   % Scaling factor (resize factor)
            %       sel.normalize           = false;% Normalize (0 mean, std = 1)
            %       sel.histeq              = false;% Histogram equalization
            %       sel.histmatch           = false;% Histogram match (ref. image: first image of the class)
            %       sel.class_range         = [];   % Class range for training database or all (tr, val, te)            
            %       sel.val_class_range     = [];   % Class range for validation database
            %       sel.te_class_range      = [];   % Class range for testing database
            %       sel.pre_process_script  = [];   % Custom preprocessing script
            % 
            %
            % Returns
            % -------
            % nndbs_tr : NNdb or cell- NNdb
            %      Training NNdb object(s). (Incase patch division is required).
            %
            % nndbs_val : NNdb or cell- NNdb
            %      Validation NNdb object(s). (Incase patch division is required).
            %
            % nndbs_te : NNdb or cell- NNdb
            %      Testing NNdb object(s). (Incase patch division is required).   
            %
            % nndbs_tr_out : NNdb or cell- NNdb
            %      Training target NNdb object(s). (Incase patch division is required).
            %
            % nndbs_val_out : NNdb or cell- NNdb
            %      Validation target NNdb object(s). (Incase patch division is required).
            %
            
            % Imports
            import nnf.db.DbSlice; 
            import nnf.db.Format;
            import nnf.pp.im_pre_process;
                        
            % Set defaults for arguments  
            if (nargin < 2 || isempty(sel))
                sel.tr_col_indices      = [1:nndb.n_per_class(1)];
                sel.tr_noise_mask       = [];
                sel.tr_noise_rate       = [];
                sel.tr_out_col_indices  = [];
                sel.val_col_indices     = [];
                sel.val_out_col_indices = [];
                sel.te_col_indices      = [];
                sel.nnpatches           = [];
                sel.use_rgb             = true;
                sel.color_index         = [];                
                sel.use_real            = false;
                sel.scale               = [];
                sel.normalize           = false;
                sel.histeq              = false;
                sel.histmatch           = false;
                sel.class_range         = [];
                sel.val_class_range     = [];
                sel.te_class_range      = [];
                sel.pre_process_script  = [];                
            end

            % Set defaults for selection fields, if the field does not exist            
            if (~isfield(sel, 'tr_noise_mask'));        sel.tr_noise_mask       = [];       end;
            if (~isfield(sel, 'tr_noise_rate'));        sel.tr_noise_rate       = [];       end;
            if (~isfield(sel, 'tr_out_col_indices'));   sel.tr_out_col_indices  = [];       end;
            if (~isfield(sel, 'val_col_indices'));      sel.val_col_indices     = [];       end;
            if (~isfield(sel, 'val_out_col_indices'));  sel.val_out_col_indices = [];       end;
            if (~isfield(sel, 'te_col_indices'));       sel.te_col_indices      = [];       end;
            if (~isfield(sel, 'nnpatches'));            sel.nnpatches           = [];       end;
            if (~isfield(sel, 'use_rgb'));              sel.use_rgb             = true;     end;
            if (~isfield(sel, 'color_index'));          sel.color_index         = [];       end;
            if (~isfield(sel, 'use_real'));             sel.use_real            = false;    end;
            if (~isfield(sel, 'scale'));                sel.scale               = [];       end;
            if (~isfield(sel, 'normalize'));            sel.normalize           = false;    end;
            if (~isfield(sel, 'histeq'));               sel.histeq              = false;    end;
            if (~isfield(sel, 'histmatch'));            sel.histmatch           = false;    end;
            if (~isfield(sel, 'class_range'));          sel.class_range         = [];       end;
            if (~isfield(sel, 'val_class_range'));      sel.val_class_range     = [];       end;
            if (~isfield(sel, 'te_class_range'));       sel.te_class_range      = [];       end;
            if (~isfield(sel, 'pre_process_script'));   sel.pre_process_script  = [];       end;
            
          	% Error handling for arguments
            if (isempty(sel.tr_col_indices) && ...
                isempty(sel.tr_out_col_indices) && ...
                isempty(sel.val_col_indices) && ...
                isempty(sel.val_out_col_indices) && ...
                isempty(sel.te_col_indices))
                error('ARG_ERR: [tr|tr_out|val|val_out|te]_col_indices: mandory field');
            end            
            if (sel.use_rgb && ~isempty(sel.color_index))
                error('ARG_CONFLICT: sel.use_rgb, sel.color_index');
            end            
            if (~isempty(sel.tr_noise_mask) && isempty(sel.tr_noise_rate))
                error('ARG_MISSING: specify sel.tr_noise_rate field');
            end
                           
            % Fetch the counts
            tr_n_per_class      = numel(sel.tr_col_indices);
            tr_out_n_per_class  = numel(sel.tr_out_col_indices);
            val_n_per_class     = numel(sel.val_col_indices);
            val_out_n_per_class = numel(sel.val_out_col_indices);
            te_n_per_class      = numel(sel.te_col_indices);

            % Set defaults for class range (tr or tr, val, te)
            cls_range           = sel.class_range;
            if (isempty(cls_range))
                cls_range = 1:nndb.cls_n;
            end   
            
            % Set defaults for other class ranges (val, te)
            val_cls_range = sel.val_class_range;
            te_cls_range = sel.te_class_range;
            if (isempty(sel.val_class_range)); val_cls_range = cls_range; end
            if (isempty(sel.te_class_range)); te_cls_range = cls_range; end
            
            % NOTE: TODO: Whitening the root db did not perform well (co-variance is indeed needed)
            
            % Initialize NNdb cell arrays
            type         = class(nndb.db);  
            nndbs_tr     = DbSlice.init_nndb('Training', type, sel, nndb, tr_n_per_class, numel(cls_range), true);
            nndbs_tr_out = DbSlice.init_nndb('Cannonical', type, sel, nndb, tr_out_n_per_class, numel(cls_range), false);
            nndbs_val    = DbSlice.init_nndb('Validation', type, sel, nndb, val_n_per_class, numel(val_cls_range), true);   
            nndbs_val_out= DbSlice.init_nndb('ValCannonical', type, sel, nndb, val_out_n_per_class, numel(val_cls_range), false);
            nndbs_te     = DbSlice.init_nndb('Testing', type, sel, nndb, te_n_per_class, numel(te_cls_range), true);
                       
            % Fetch iterative range
            data_range = DbSlice.get_data_range(cls_range, val_cls_range, te_cls_range, ...
                            sel.tr_col_indices, sel.tr_out_col_indices, ...
                            sel.val_col_indices, sel.val_out_col_indices, ...
                            sel.te_col_indices, nndb);
           
            % Iterate over the cls_st indices (i_cls => current cls index)
            i_cls = 1;
            
            % Patch count
            patch_loop_max_n = numel(sel.nnpatches);
            if (patch_loop_max_n == 0); patch_loop_max_n = 1; end;
                
            % Initialize the indices
            tr_idxs = uint16(ones(1, patch_loop_max_n));
            tr_out_idxs = uint16(ones(1, patch_loop_max_n));
            val_idxs = uint16(ones(1, patch_loop_max_n));
            val_out_idxs = uint16(ones(1, patch_loop_max_n));
            te_idxs = uint16(ones(1, patch_loop_max_n));
            
            % PERF: Noise required indices (avoid find in each iteration)
            nf = find(sel.tr_noise_mask == 1);
          
            % Iterate through images in nndb
            for i=data_range
         
                % Update the current prev_cls_en
                % Since 'i' may not be consecutive
                while ((numel(nndb.cls_st) >= (i_cls+1)) && (i >= nndb.cls_st(i_cls+1)))
                    i_cls = i_cls + 1;
                end
                prev_cls_en = nndb.cls_st(i_cls) - 1;
                
                % Checks whether current 'img' needs processing
                p = ~DbSlice.dicard_needed(nndbs_tr, i, i_cls, cls_range, prev_cls_en, sel.tr_col_indices);
                p = p | ~DbSlice.dicard_needed(nndbs_tr_out, i, i_cls, cls_range, prev_cls_en, sel.tr_out_col_indices);
                p = p | ~DbSlice.dicard_needed(nndbs_val, i, i_cls, val_cls_range, prev_cls_en, sel.val_col_indices);
                p = p | ~DbSlice.dicard_needed(nndbs_val_out, i, i_cls, val_cls_range, prev_cls_en, sel.val_out_col_indices);
                p = p | ~DbSlice.dicard_needed(nndbs_te, i, i_cls, te_cls_range, prev_cls_en, sel.te_col_indices);
                if (~p); continue; end
                                    
                % Color image
                cimg = nndb.get_data_at(i); 
                
                
                % Iterate through image patches
                for pI=1:patch_loop_max_n 
                    
                    % Holistic image (by default)
                    img = cimg;
                    
                    % Init variables
                    if (~isempty(sel.nnpatches))                        
                        nnpatch = sel.nnpatches(pI);
                        x = nnpatch.offset(2);
                        y = nnpatch.offset(1);                                                                                
                        w = nnpatch.w;
                        h = nnpatch.h;
                        
                        % Extract the patch
                        img = cimg(y:y+h-1, x:x+w-1, :);                        
                    end
                    
                    % Peform image operations only if db format comply them
                    if (nndb.format == Format.H_W_CH_N)

                        % Perform resize
                        if (~isempty(sel.scale))
                            img = imresize(img, sel.scale);     
                        end

                        % Perform histrogram matching against the cannonical image   
                        cls_st_img        = [];
                        if (sel.histmatch)
                            if (~isempty(sel.scale))              
                                cls_st = prev_cls_en + 1;                     
                                cls_st_img = imresize(nndb.get_data_at(cls_st), selection.scale);
                            end
                        end

                        % Color / Gray Scale Conversion (if required) 
                        img = DbSlice.process_color(img, sel);
                        cls_st_img = DbSlice.process_color(cls_st_img, sel);

                        % Pre-Processing
                        pp_params.histeq       = sel.histeq;
                        pp_params.normalize    = sel.normalize;
                        pp_params.histmatch    = sel.histmatch;
                        pp_params.cann_img     = cls_st_img;
                        img = im_pre_process(img, pp_params);

                        % [CALLBACK] the specific pre-processing script
                        if (~isempty(sel.pre_process_script))               
                            img = sel.pre_process_script(img); 
                        end                                       
                    end
                        
                    % Build Training DB
                    [nndbs_tr, tr_idxs] = ...
                        DbSlice.build_nndb_tr(nndbs_tr, pI, tr_idxs, i, i_cls, cls_range, prev_cls_en, img, sel, nf);

                    % Build Training Target DB
                    [nndbs_tr_out, tr_out_idxs] = ...
                        DbSlice.build_nndb_tr_out(nndbs_tr_out, pI, tr_out_idxs, i, i_cls, cls_range, prev_cls_en, img, sel);

                    % Build Valdiation DB
                    [nndbs_val, val_idxs] = ...
                        DbSlice.build_nndb_val(nndbs_val, pI, val_idxs, i, i_cls, val_cls_range, prev_cls_en, img, sel);

                    % Build Valdiation Target DB
                    [nndbs_val_out, val_out_idxs] = ...
                        DbSlice.build_nndb_val_out(nndbs_val_out, pI, val_out_idxs, i, i_cls, val_cls_range, prev_cls_en, img, sel);
                    
                    % Build Testing DB
                    [nndbs_te, te_idxs] = ...
                        DbSlice.build_nndb_te(nndbs_te, pI, te_idxs, i, i_cls, te_cls_range, prev_cls_en, img, sel);
                end
            end              
            
            % Returns NNdb object instead of cell array (non patch requirement)
            if (isempty(sel.nnpatches))               
                if (~isempty(nndbs_tr)); nndbs_tr = nndbs_tr{1}; end
                if (~isempty(nndbs_val)); nndbs_val = nndbs_val{1}; end                
                if (~isempty(nndbs_te)); nndbs_te = nndbs_te{1}; end
                if (~isempty(nndbs_tr_out)); nndbs_tr_out = nndbs_tr_out{1}; end
                if (~isempty(nndbs_val_out)); nndbs_val_out = nndbs_val_out{1}; end
                if (isempty(nndbs_tr)); nndbs_tr = []; end
                if (isempty(nndbs_val)); nndbs_val = []; end                
                if (isempty(nndbs_te)); nndbs_te = []; end
                if (isempty(nndbs_tr_out)); nndbs_tr_out = []; end
                if (isempty(nndbs_val_out)); nndbs_val_out = []; end  
            end                
        end       
   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function examples(imdb_8)
            %EXAMPLES: Extensive example set
            %   Assume:
            %   There are only 8 images per subject in the database. 
            %   NNdb is in H x W x CH x N format. (image database)
            %
            
            %%% Full set of options
            % nndb = NNdb('original', imdb_8, 8, true);
            % sel.tr_col_indices        = [1:3 7:8]; %[1 2 3 7 8]; 
            % sel.tr_noise_mask         = [];
            % sel.tr_noise_rate         = [];
            % sel.tr_out_col_indices    = [];
            % sel.tr_cm_col_indices     = [];
            % sel.te_col_indices        = [4:6]; %[4 5 6]
            % sel.use_rgb               = false;
            % sel.color_index           = [];                
            % sel.use_real              = false;
            % sel.scale                 = 0.5;
            % sel.normalize             = false;
            % sel.histeq                = true;
            % sel.histmatch             = false;
            % sel.class_range           = [1:36 61:76 78:100];
            % %sel.pre_process_script    = @custom_pprocess;
            % sel.pre_process_script    = [];
            % [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel); 
            
            % Imports
            import nnf.db.NNdb;
            import nnf.db.DbSlice;
            import nnf.core.generators.NNPatchGenerator;
            
            
            % 
            % Select 1st 2nd 4th images of each identity for training.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4]; %[1 2 4]; 
            [nndb_tr, ~, ~, ~] = DbSlice.slice(nndb, sel); % nndb_tr = DbSlice.slice(nndb, sel); 

            
            %
            % Select 1st 2nd 4th images of each identity for training.
            % Divide into patches
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4]; %[1 2 4];             
            patch_gen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33);
            sel.nnpatches = patch_gen.generate_patches();

            % Cell arrays of NNdb objects for each patch
            [nndbs_tr, ~, ~, ~] = DbSlice.slice(nndb, sel);
            nndbs_tr{1}.show()
            figure, 
            nndbs_tr{2}.show()
            figure, 
            nndbs_tr{3}.show()
            figure,
            nndbs_tr{4}.show() 
            
            
            % 
            % Select 1st 2nd 4th images of each identity for training.
            % Select 3rd 5th images of each identity for testing.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
           	% 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 3rd 5th images of identities denoted by class_range for testing.            
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 1st 2nd 4th images images of identities denoted by class_range for validation.   
            % Select 3rd 5th images of identities denoted by class_range for testing. 
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.val_col_indices= [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            [nndb_tr, nndb_val, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 1st 2nd 4th images images of identities denoted by val_class_range for validation.   
            % Select 3rd 5th images of identities denoted by te_class_range for testing. \
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.val_col_indices= [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            sel.val_class_range= [6:15];
            sel.te_class_range = [17:20];
            [nndb_tr, nndb_val, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 3rd 4th images of identities denoted by val_class_range for validation.
            % Select 3rd 5th images of identities denoted by te_class_range for testing.
            % Select 1st 1st 1st images of identities denoted by class_range for training target.
            % Select 1st 1st images of identities denoted by val_class_range for validation target.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices      = [1:2 4];
            sel.val_col_indices     = [3 4];
            sel.te_col_indices      = [3 5]; 
            sel.tr_out_col_indices  = [1 1 1];
            sel.val_out_col_indices = [1 1];  
            sel.class_range         = [1:10];
            sel.val_class_range     = [6:15];
            sel.te_class_range      = [17:20];
            [nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out] = DbSlice.slice(nndb, sel);
            nndb_tr.show(10, 3)
            figure, nndb_val.show(10, 2)
            figure, nndb_te.show(4, 2)
            figure, nndb_tr_out.show(10, 3)            
            figure, nndb_val_out.show(10, 2) 

            
            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add various noise types @ random locations of varying degree.
            %               default noise type: random black and white dots.
            % Select 3rd 5th images of each identity for testing.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.tr_noise_mask  = [0 1 0 1 0 1];             % index is affected or not
            sel.tr_noise_rate  = [0 0.5 0 0.5 0 0.5];       % percentage of corruption
            %sel.tr_noise_rate  = [0 0.5 0 0.5 0 Noise.G];  % last index with Gauss noise
            sel.tr_col_indices = [1:2 4];   %[1 2 4];
            sel.te_col_indices = [3 5];     %[3 5]; 
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
           	% 
            % To prepare regression datasets, training dataset and training target dataset
            % Select 1st 2nd 4th images of each identity for training.
            % Select 1st 1st 1st image of each identity for corresponding training target.
            % Select 3rd 5th images of each identity for testing.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1 2 4];       %[1 2 4]; 
            sel.tr_out_col_indices = [1 1 1];   %[1 1 1]; (Regression 1->1, 2->1, 4->1) 
            sel.te_col_indices = [3 5];         %[3 5]; 
            [nndb_tr, nndb_tr_out, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            %  
            % Resize images by 0.5 scale factor.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.scale          = 0.5;
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
            %  
            % Use gray scale images.
            % Perform histrogram equalization.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.histeq         = true;
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            
            %  
            % Use gray scale images.
            % Perform histrogram match. This will be performed with the 1st image of each identity
            % irrespective of the selection choice. (refer code for more details)
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.histmatch      = true;
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
            %
            % If imdb_8 supports many color channels
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.color_index    = 5;         % color channel denoted by 5th index
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
	methods (Access = private, Static) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [new_nndbs] = init_nndb(name, type, sel, nndb, n_per_class, cls_n, build_cls_idx) 
            % INIT_NNDB: inits a new NNdb object cell array.
            %            
            % Returns
            % -------
            % new_nndbs : cell- NNdb
            %      new NNdb object(s). 
            %
            
            % Imports
            import nnf.db.NNdb; 
            import nnf.db.Format;
            
            new_nndbs = cell(0, 1);
            
            % n_per_class: must be a scalar for this method
            if (n_per_class == 0); return; end;
                
            % Init Variables
            db = cell(0, 0);
                
            % Peform image operations only if db format comply them
            if (nndb.format == Format.H_W_CH_N)
            
                % Dimensions for the new NNdb Objects
                nd1 = nndb.h;
                nd2 = nndb.w; 

                if (~isempty(sel.scale))
                    nd1 = nd1 * sel.scale;
                    nd2 = nd2 * sel.scale;   
                end
                    
                % Channels for the new NNdb objects
                ch = nndb.ch;
                if (~(sel.use_rgb))
                    ch = numel(sel.color_index); % Selected color channels
                    if (ch == 0); ch = 1; end % Grayscale
                end                
                
                if (~isempty(sel.nnpatches))                    
                    % Patch count                    
                    patch_n = numel(sel.nnpatches);                    
                    
                    % Init db for each patch
                    for i=1:patch_n
                        nnpatch = sel.nnpatches(i);
                        db{i} = cast(zeros(nnpatch.h, nnpatch.w, ch, n_per_class*cls_n), type);
                    end                                                     
            
                else
                    db{1} = cast(zeros(nd1, nd2, ch, n_per_class*cls_n), type);
                    
                end
                    
            
            elseif (nndb.format == Format.H_N)
                nd1 = nndb.h;
                db{1} = cast(zeros(nd1, n_per_class*cls_n), type);
            
                % TODO: implement patch support                        
            end
            
            % Init nndb for each patch
            for i=1:numel(db)
                new_nndbs{i} = NNdb(name, db{i}, n_per_class, build_cls_idx); 
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data_range = get_data_range(cls_range, val_cls_range, te_cls_range, ...
                                    tr_col, tr_out_col, val_col, val_out_col, te_col, nndb) 
            % GET_DATA_RANGE: fetches the data_range (images indices).          
            %            
            % Returns
            % -------
            % img : vector -uint32
            %      data indicies.
            %
            
            % Union of all class ranges
            cls_range = union(union(cls_range, val_cls_range), te_cls_range);
            
            % Union of all col indices
            col_indices = union(union(union(union(tr_col, tr_out_col), val_col), val_out_col), te_col);
            
            % Total class count
            cls_n = numel(cls_range);
            
            % *Ease of implementation
            % Allocate more memory, shrink it later
            data_range = uint32(zeros(1, cls_n * numel(col_indices)));   
                       
            st = 1;
            for i = 1:cls_n
                ii = cls_range(i);
                dst = nndb.cls_st(ii);
                
                data_range(st:st+numel(col_indices)-1) = uint32(col_indices) + (dst-1);               
                st = st + numel(col_indices);              
            end
            
            % Shrink the vector (Can safely ignore this code)
            data_range(st:end) = [];
        end
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [img] = process_color(img, sel) 
            % PROCESS_COLOR: performs color related functions.
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
                    if (numel(sel.color_index) > 0)
                        X = img(:, :, sel.color_index);                       
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
        function [nndbs, ni] = build_nndb_tr(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel, noise_found) 
            % BUILD_NNDB_TR: builds the nndb training database.
            %          
            % Returns
            % -------
            % nndbs : cell- NNdb
            %      Updated NNdb objects.
            %
            % ni : vector -uint16
            %      Updated image count vector.
            %  
            
            % Imports 
            import nnf.db.DbSlice;
            import nnf.db.Format;
            import nnf.db.Noise;
             
            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices             
            [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.tr_col_indices);               
            if (discard); return; end; 
            
            % Iterate over found indices
            for j=1:numel(found)
                    
                % Check whether found contain a noise required index
                if (find(noise_found == found(j)))
                                        
                    % Currently supports noise for images only
                    if (nndb.format ~= Format.H_W_CH_N)
                        nndb.set_data_at(img, ni(pi)); 
                        continue;
                    end
                                        
                    [h, w, ch] = size(img);

                    % Fetch noise rate
                    rate = sel.tr_noise_rate(found(j)); 

                    % Add different noise depending on the type or rate
                    % (ref. Enums/Noise)
                    if (rate == Noise.G)
                        img = imnoise(img, 'gaussian');
%                             img = imnoise(img, 'gaussian');
%                             img = imnoise(img, 'gaussian');
%                             img = imnoise(img, 'gaussian');

                    else
                        % Perform random corruption
                        % Corruption Size (H x W)
                        cs = [uint16(h*rate) uint16(w*rate)]; 

                        % Random location choice
                        % Start of H, W (location)
                        sh = 1 + rand()*(h-cs(1)-1);
                        sw = 1 + rand()*(w-cs(2)-1);

                        % Set the corruption
                        cimg = uint8(DbSlice.rand_corrupt(cs(1), cs(2)));

                        if (ch == 1)
                            img(sh:sh+cs(1)-1, sw:sw+cs(2)-1) = cimg;
                        else
                            for ich=1:ch
                                img(sh:sh+cs(1)-1, sw:sw+cs(2)-1, ich) = cimg;
                            end
                        end
                    end

                    nndb.set_data_at(img, ni(pi));
                else
                    nndb.set_data_at(img, ni(pi));
                end

                ni(pi) = ni(pi) + 1;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function img = rand_corrupt(height, width) 
            % RAND_CORRUPT: corrupts the image with a (height, width) block.
            %            
            % Returns
            % -------
            % img : 3D tensor -uint8
            %      Corrupted image.
            %
            
            percentageWhite = 50; % Alter this value as desired

            dotPattern = zeros(height, width);

            % Set the desired percentage of the elements in dotPattern to 1
            dotPattern(1:round(0.01*percentageWhite*numel(dotPattern))) = 1;

            % Seed the random number generator
%             rand('twister',100*sum(clock));

            % Randomly permute the element order
            dotPattern = reshape(dotPattern(randperm(numel(dotPattern))), height, width);
            img = dotPattern .* 255;

            % imagesc(dotPattern);
            % colormap('gray');
            % axis equal;

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, ni] = build_nndb_tr_out(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel) 
            % BUILD_NNDB_TR_OUT: builds the nndb training target database.
            %            
            % Returns
            % -------
            % nndbs : cell- NNdb
            %      Updated NNdb objects.
            %
            % ni : vector -uint16
            %      Updated image count vector.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.tr_out_col_indices);                
            if (discard); return; end; 
            
            % Iterate over found indices
            for j=1:numel(found)
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(pi) + 1;
            end 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, ni] = build_nndb_val(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel) 
            % BUILD_NNDB_VAL: builds the nndb validation database.
            %            
            % Returns
            % -------
            % nndbs : cell- NNdb
            %      Updated NNdb objects.
            %
            % ni : vector -uint16
            %      Updated image count vector.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.val_col_indices);                
            if (discard); return; end; 

            % Iterate over found indices
            for j=1:numel(found)
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(pi) + 1;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, ni] = build_nndb_val_out(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel) 
            % BUILD_NNDB_VAL_OUT: builds the nndb validation target database.
            %            
            % Returns
            % -------
            % nndbs : cell- NNdb
            %      Updated NNdb objects.
            %
            % ni : vector -uint16
            %      Updated image count vector.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.val_out_col_indices);                
            if (discard); return; end; 

            % Iterate over found indices
            for j=1:numel(found)
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(pi) + 1;
            end
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, ni] = build_nndb_te(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel) 
            % BUILD_NNDB_TE: builds the testing database.
            %            
            % Returns
            % -------
            % nndbs : cell- NNdb
            %      Updated NNdb objects.
            %
            % ni : vector -uint16
            %      Updated image count vector.
            %
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.te_col_indices);                
            if (discard); return; end;  
                    
            % Iterate over found indices
            for j=1:numel(found) 
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(pi) + 1;
            end 
        end
        
      	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [found] = find(i, prev_cls_en, col_indices) 
            % FIND: checks whether 'i' is in required indices

            found = [];             
            if (isempty(col_indices)); return; end;

            % Find whether 'i' is in required indices
            found = find((mod(double(i)-double(col_indices), double(prev_cls_en)) == 0) == 1);  
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [discard, found] = dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, col_indices) 
            % DICARD_NEEDED: checks whether the given index 'i' needs be dicarded.
            %
            % Returns
            % -------
            % discard : bool
            %      Boolean indicating the dicard. 
            %
            % found : bool
            %      If discard=false, found denotes the index 'i' found positions in col_indices.
            %
            
            % Imports 
            import nnf.db.DbSlice;
            
            found = DbSlice.find(i, prev_cls_en, col_indices);            
            discard = ~(~isempty(nndbs) && ...
                        ~isempty(intersect(i_cls, cls_range)) && ...
                        ~isempty(found));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
            



        


