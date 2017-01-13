classdef DbSlice 
    % DBSLICE peforms slicing of nndb with the help of a selection structure.
    % IMPL_NOTES: Static class (thread-safe).
    %
    % Selection Structure (with defaults)
    % -----------------------------------
    % sel.tr_col_indices    = [];   % Training column indices
    % sel.tr_noise_mask     = [];   % Noisy tr. col indices (bit mask)
    % sel.tr_noise_rate     = [];   % Rate or noise types for the above field
    % sel.tr_out_col_indices= [];   % Training target column indices
    % sel.tr_cm_col_indices = [];   % TODO: Document
    % sel.te_col_indices    = [];   % Testing column indices
    % sel.use_rgb           = true; % Use rgb or convert to grayscale
    % sel.color_index       = [];   % Specific color indices (set .use_rgb = false)             
    % sel.use_real          = false;% Use real valued database TODO: (if .normalize = true, Operations ends in real values)
    % sel.scale             = [];   % Scaling factor (resize factor)
    % sel.normalize         = false;% Normalize (0 mean, std = 1)
    % sel.histeq            = false;% Histogram equalization
    % sel.histmatch         = false;% Histogram match (ref. image: first image of the class)
    % sel.class_range       = [];   % Class range
    % sel.pre_process_script= [];   % Custom preprocessing script
    %
    % i.e
    % Pass a selection structure to split the nndb accordingly
    % [nndb_tr, ~, ~, ~] = DbSlice.slice(nndb, sel);
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
        
    methods (Access = public, Static) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs_tr, nndbs_tr_out, nndbs_te, nndbs_tr_cm] = slice(nndb, sel) 
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
            %       sel.tr_col_indices    = [];   % Training column indices
            %       sel.tr_noise_mask     = [];   % Noisy tr. col indices (bit mask)
            %       sel.tr_noise_rate     = [];   % Rate or noise types for the above field
            %       sel.tr_out_col_indices= [];   % Training target column indices
            %       sel.tr_cm_col_indices = [];   % TODO: Document
            %       sel.te_col_indices    = [];   % Testing column indices
            %       sel.use_rgb           = true; % Use rgb or convert to grayscale
            %       sel.color_index       = [];   % Specific color indices (set .use_rgb = false)             
            %       sel.use_real          = false;% Use real valued database TODO: (if .normalize = true, Operations ends in real values)
            %       sel.scale             = [];   % Scaling factor (resize factor)
            %       sel.normalize         = false;% Normalize (0 mean, std = 1)
            %       sel.histeq            = false;% Histogram equalization
            %       sel.histmatch         = false;% Histogram match (ref. image: first image of the class)
            %       sel.class_range       = [];   % Class range
            %       sel.pre_process_script= [];   % Custom preprocessing script
            % 
            %
            % Returns
            % -------
            % nndbs_tr : cell- NNdb
            %      Training datasets (Incase patch division is required).
            %
            % nndbs_tr_out : cell- NNdb
            %      Training target datasets (Incase patch division is required).
            %
            % nndbs_te : cell- NNdb
            %      Testing target datasets (Incase patch division is required).
            %
            % nndbs_tr_cm : cell- NNdb
            %      TODO: Yet to be documenteds (Incase patch division is required).
            
            % Imports
            import nnf.db.DbSlice; 
            import nnf.db.Format;
            import nnf.pp.im_pre_process;
                        
            % Set defaults for arguments  
            if (nargin < 2 || isempty(sel))
                sel.tr_col_indices        = [1:nndb.n_per_class(1)];
                sel.tr_noise_mask         = [];
                sel.tr_noise_rate         = [];
                sel.tr_out_col_indices    = [];
                sel.tr_cm_col_indices     = [];
                sel.te_col_indices        = [];
                sel.nnpatches             = [];
                sel.use_rgb               = true;
                sel.color_index           = [];                
                sel.use_real              = false;
                sel.scale                 = [];
                sel.normalize             = false;
                sel.histeq                = false;
                sel.histmatch             = false;
                sel.class_range           = [];
                sel.pre_process_script    = [];                
            end

            % Set defaults for selection fields, if the field does not exist            
            if (~isfield(sel, 'tr_noise_mask'));              sel.tr_noise_mask     = [];       end;
            if (~isfield(sel, 'tr_noise_rate'));              sel.tr_noise_rate     = [];       end;
            if (~isfield(sel, 'tr_out_col_indices'));         sel.tr_out_col_indices= [];       end;
            if (~isfield(sel, 'tr_cm_col_indices'));          sel.tr_cm_col_indices = [];       end;
            if (~isfield(sel, 'te_col_indices'));             sel.te_col_indices    = [];       end;
            if (~isfield(sel, 'nnpatches'));                  sel.nnpatches         = [];       end;
            if (~isfield(sel, 'use_rgb'));                    sel.use_rgb           = true;     end;
            if (~isfield(sel, 'color_index'));                sel.color_index       = [];       end;
            if (~isfield(sel, 'use_real'));                   sel.use_real          = false;    end;
            if (~isfield(sel, 'scale'));                      sel.scale             = [];       end;
            if (~isfield(sel, 'normalize'));                  sel.normalize         = false;    end;
            if (~isfield(sel, 'histeq'));                     sel.histeq            = false;    end;
            if (~isfield(sel, 'histmatch'));                  sel.histmatch         = false;    end;
            if (~isfield(sel, 'class_range'));                sel.class_range       = [];       end;
            if (~isfield(sel, 'pre_process_script'));         sel.pre_process_script= [];       end;
            
          	% Error handling for arguments
            if (isempty(sel.tr_col_indices) && isempty(sel.tr_out_col_indices) ...
                && isempty(sel.tr_cm_col_indices) && isempty(sel.te_col_indices))
                error('ARG_ERR: [tr|tr_out|tr_cm|te]_col_indices: mandory field');
            end            
            if (sel.use_rgb && ~isempty(sel.color_index))
                error('ARG_CONFLICT: sel.use_rgb, sel.color_index');
            end            
            if (~isempty(sel.tr_noise_mask) && isempty(sel.tr_noise_rate))
                error('ARG_MISSING: specify sel.tr_noise_rate field');
            end
                           
            % Fetch the counts
            tr_n_per_class         = numel(sel.tr_col_indices);
            tr_out_n_per_class     = numel(sel.tr_out_col_indices);
            tr_cm_n_per_class      = numel(sel.tr_cm_col_indices);
            te_n_per_class         = numel(sel.te_col_indices);

            cls_range              = sel.class_range;
            if (isempty(cls_range))
                cls_range = 1:nndb.cls_n;
            end           

            % NOTE: TODO: Whitening the root db did not perform well (co-variance is indeed needed)
            
            % Initialize NNdb Objects
            type         = class(nndb.db);
            cls_n        = numel(cls_range);  
            nndbs_tr     = DbSlice.init_nndb('Training', type, sel, nndb, tr_n_per_class, cls_n, true);
            nndbs_te     = DbSlice.init_nndb('Testing', type, sel, nndb, te_n_per_class, cls_n, true);
            nndbs_tr_out = DbSlice.init_nndb('Cannonical', type, sel, nndb, tr_out_n_per_class, cls_n, false);
            nndbs_tr_cm  = DbSlice.init_nndb('Cluster-Means', type, sel, nndb, tr_cm_n_per_class, cls_n, false);            
                       
            % Fetch iterative range
            data_range = DbSlice.get_data_range(cls_range, nndb);
           
            % Iterate over the cls_st indices
            j = 1;
            
            % Patch count
            patch_n = numel(sel.nnpatches);    
                
            % Initialize the indices
            tr_idxs = uint16(ones(1, patch_n));
            tr_out_idxs = uint16(ones(1, patch_n));
            tr_cm_idxs = uint16(ones(1, patch_n));
            te_idxs = uint16(ones(1, patch_n));
            
            % PERF: Noise required indices (avoid find in each iteration)
            nf = find(sel.tr_noise_mask == 1);
          
            % Iterate through images in nndb
            for i=data_range
                
                % Colored image
                cimg = nndb.get_data_at(i);                   
                            
                % Update the current prev_cls_en
                % Since 'i' may not be consecutive
                while ((numel(nndb.cls_st) >= (j+1)) && (i >= nndb.cls_st(j+1)))
                    j = j + 1;
                end
                prev_cls_en = nndb.cls_st(j) - 1;
                
                % Checks whether current 'img' needs processing
                f = (~isempty(nndbs_tr) && ~isempty(DbSlice.find(i, prev_cls_en, sel.tr_col_indices)));
                f = f | (~isempty(nndbs_tr_out) && ~isempty(DbSlice.find(i, prev_cls_en, sel.tr_out_col_indices)));
                f = f | (~isempty(nndbs_tr_cm) && ~isempty(DbSlice.find(i, prev_cls_en, sel.tr_cm_col_indices)));
                f = f | (~isempty(nndbs_te) && ~isempty(DbSlice.find(i, prev_cls_en, sel.te_col_indices)));
                if (~f); continue; end
                    
                % Divide the img into patches
                for pI=1:patch_n
                    
                    % Init variables
                    nnpatch = sel.nnpatches(pI);
                    x = nnpatch.offset(1);
                    y = nnpatch.offset(2);                                                            
                    w = nnpatch.w;
                    h = nnpatch.h;
                    
                    % Extract the patch
                    img = cimg(y:y+h-1, x:x+w-1, :);
                
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
                        DbSlice.build_nndb_tr(nndbs_tr, pI, tr_idxs, i, prev_cls_en, img, sel, nf);

                    % Build Training Output DB
                    [nndbs_tr_out, tr_out_idxs] = ...
                        DbSlice.build_nndb_tr_out(nndbs_tr_out, pI, tr_out_idxs, i, prev_cls_en, img, sel);

                    % Build Training Cluster Centers (For Multi Label DDA) DB
                    [nndbs_tr_cm, tr_cm_idxs] = ...
                        DbSlice.build_nndb_tr_cm(nndbs_tr_cm, pI, tr_cm_idxs, i, prev_cls_en, img, sel);

                    % Build Testing DB
                    [nndbs_te, te_idxs] = ...
                        DbSlice.build_nndb_te(nndbs_te, pI, te_idxs, i, prev_cls_en, img, sel);
                end
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
            
            % 
            % Select 1st 2nd 4th images of each identity for training.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = [];
            sel.tr_col_indices = [1:2 4]; %[1 2 4]; 
            [nndb_tr, ~, ~, ~] = DbSlice.slice(nndb, sel); % nndb_tr = DbSlice.slice(nndb, sel); 
            
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
            sel.class_range    = [1:10];    % First then identities 
            [nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb, sel);
            
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
            % INIT_NNDB: inits a new empty 4D tensor database with a nndb.
 
            % Imports
            import nnf.db.NNdb; 
            import nnf.db.Format;
            
            new_nndbs = cell(0, 1);
            
            % n_per_class: must be a scalar
            if (n_per_class == 0); return; end;
                           
            % Peform image operations only if db format comply them
            if (nndb.format == Format.H_W_CH_N || nndb.format == H_W_CH_N_NP)
            
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
                
                % Init Variables
                db = cell(0, 1);
                
                if (~isempty(sel.nnpatches))
%                     if (nndb.format == H_W_CH_N_NP); %  TODO: Remove type H_W_CH_N_NP
%                         error('COMP_ERR: nndb is already in H_W_CH_N_NP format.');
%                     end   
                    
                    % Patch count                    
                    patch_n = numel(sel.nnpatches);                    
                    
                    % Init db for each patch
                    for i=1:patch_n
                        nnpatch = sel.nnpatches(i);
                        db{i} = cast(zeros(nnpatch.h, nnpatch.w, ch, n_per_class*cls_n), type);
                    end                                                     
            
                elseif (nndb.format == Format.H_W_CH_N)
                    db{1} = cast(zeros(nd1, nd2, ch, n_per_class*cls_n), type);
                    
                else % (nndb.format == Format.H_W_CH_N_NP) TODO: Remove type H_W_CH_N_NP
                    assert(false); % TODO: Implement
                end
                    
            
            elseif (nndb.format == Format.H_N)
                nd1 = nndb.h;
                db{1} = cast(zeros(nd1, n_per_class*cls_n), type);
            
                % TODO: implement patch support
                
            elseif (nndb.format == Format.H_N_NP)
                nd1 = nndb.h;
                assert(false); % TODO: Implement
                db{1} = cast(zeros(nd1, n_per_class*cls_n, nndb.p), type);                
            end
            
            % Init nndb for each patch
            for i=1:numel(db)
                new_nndbs{i} = NNdb(name, db{i}, n_per_class, build_cls_idx); 
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data_range = get_data_range(cls_range, nndb) 
            % GET_DATA_RANGE: fetches the data_range (images indices).
            
            % Class count
            cls_n = numel(cls_range);
            
            % *Ease of implementation
            % Allocate more memory, shrink it later
            data_range = uint32(zeros(1, cls_n * max(nndb.n_per_class)));   
                       
            % TODO: Compatibility for other NNdb Formats. (Hx W x CH x N x NP)
            st = 1;
            for i = 1:cls_n
                ii = cls_range(i);
                dst = nndb.cls_st(ii);
                data_range(st:st+nndb.n_per_class(ii)-1) = ...
                    dst:dst + uint32(nndb.n_per_class(ii)) - 1;                
                st = st + nndb.n_per_class(ii);                
            end
            
            % Shrink the vector
            data_range(st:end) = [];
        end
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [img] = process_color(img, sel) 
            % PROCESS_COLOR: performs color related functions.

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
        function [nndbs, ni] = build_nndb_tr(nndbs, pi, ni, i, prev_cls_en, img, sel, noise_found) 
            % BUILD_NNDB_TR: builds the nndb training database.
                       
            % Imports 
            import nnf.db.DbSlice;
            import nnf.db.Format;
            import nnf.db.Noise;
             
            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices 
            found = DbSlice.find(i, prev_cls_en, sel.tr_col_indices);                
            if (~found); return; end; 
            
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
        function [nndbs, ni] = build_nndb_tr_out(nndbs, pi, ni, i, prev_cls_en, img, sel) 
            % BUILD_NNDB_TR_OUT: builds the nndb training target database.
        
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            found = DbSlice.find(i, prev_cls_en, sel.tr_out_col_indices);                
            if (~found); return; end; 
            
            % Iterate over found indices
            for j=1:numel(found)
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(p1) + 1;
            end 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, ni] = build_nndb_tr_cm(nndbs, pi, ni, i, prev_cls_en, img, sel) 
            % BUILD_NNDB_TR_CM: builds the nndb training  mean centre database.
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            found = DbSlice.find(i, prev_cls_en, sel.tr_cm_col_indices);                
            if (~found); return; end; 

            % Iterate over found indices
            for j=1:numel(found)
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(pi) + 1;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs, ni] = build_nndb_te(nndbs, pi, ni, i, prev_cls_en, img, sel) 
            % BUILD_NNDB_TE: builds the testing database.
            
            % Imports 
            import nnf.db.DbSlice;

            if (isempty(nndbs)); return; end;
            nndb = nndbs{pi};
            
            % Find whether 'i' is in required indices                
            found = DbSlice.find(i, prev_cls_en, sel.te_col_indices);                
            if (~found); return; end;  
                    
            % Iterate over found indices
            for j=1:numel(found)
                nndb.set_data_at(img, ni(pi));
                ni(pi) = ni(pi) + 1;
            end 
        end
        
      	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [found] = find(i, prev_cls_en, col_indices)
            % FIND: checks whether 'i' is in required indices
           
            % Find whether 'i' is in required indices
            found = find((mod(double(i)-double(col_indices), double(prev_cls_en)) == 0) == 1);  
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
            



        


