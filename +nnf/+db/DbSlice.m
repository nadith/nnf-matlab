classdef DbSlice
    % DBSLICE peforms slicing of nndb with the help of a selection structure.
    % IMPL_NOTES: Static class (thread-safe).
    %
    % Selection Structure (with defaults)
    % -----------------------------------
    % sel.tr_col_indices        = [];   % Training column indices
    % sel.tr_noise_rate         = [];   % Noise rate or Noise types for `tr_col_indices`
    % sel.tr_occlusion_rate     = [];   % Occlusion rate for `tr_col_indices`
    % sel.tr_occlusion_type     = [];   % ('t':top, 'b':bottom, 'l':left, 'r':right) for `tr_col_indices`
    % sel.tr_occlusion_offset   = [];   % Occlusion start offset from top/bottom/left/right corner depending on `tr_occlusion_type`
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
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function safe_slice(self, nndb, sel)
            % Thread Safe
            % Imports
            import nnf.core.iters.memory.DskmanMemDataIterator;
            DbSlice.slice(nndb, sel, DskmanMemDataIterator(nndb))
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
        
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndbs_tr, nndbs_val, nndbs_te, nndbs_tr_out, nndbs_val_out, nndbs_te_out, edatasets] = ...
                                                                    slice(nndb, sel, data_generator, pp_param) 
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
                error('ARG_ERR: [tr|tr_out|val|val_out|te]_col_indices: mandatary field');
            end            
            if ((~isempty(sel.use_rgb) && sel.use_rgb) && ~isempty(sel.color_indices))
                error('ARG_CONFLICT: sel.use_rgb, sel.color_indices');
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
            % DEPENDANCY -The order must be preserved as per the enum Dataset 
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
            data_generator.init(cls_ranges, col_ranges, true);       

            % LIMITATION: PYTHON-MATLAB (no support for generators)
            % [PERF] Iterate through the choset subset of the nndb database
            [cimg, ~, cls_idx, col_idx, datasets, stop] = data_generator.next();
            while (~stop)

                % Perform pre-processing before patch division
                % Peform image operations only if db format comply them
                if (nndb.format == Format.H_W_CH_N || nndb.format == Format.N_H_W_CH)

                    % For histogram equalizaion operation (cannonical image)
                    cann_cimg = [];
                    if (~isempty(sel.histmatch_col_index))
                        [cann_cimg, ~] = data_generator.get_cimg_frecord_in_next(cls_idx, sel.histmatch_col_index);
                        % Alternative:
                        % cls_st = nndb.cls_st(sel.histmatch_col_index)
                        % cann_cimg = nndb.get_data_at(cls_st)
                    end

                    % Peform image preocessing
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
                        elseif (nndb.format == Format.H_W_CH_N || nndb.format == Format.N_H_W_CH)
                            pimg = cimg(y:y+h-1, x:x+w-1, :);
                        
                        elseif (nndb.format == Format.H_N || nndb.format == Format.N_H)
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
                                nndbs(end+1) = NNdb([Dataset.str(edataset) '_p' num2str(pi_tmp)], [], [], false, [], nndb.format);
                            end

                            % Update the dict_nndbs
                            dict_nndbs(uint32(edataset)) = nndbs;
                        end    

                        % Build Training DB
                        if (edataset == Dataset.TR)
                            
                            % If noise or occlusion is required
                            if ((~isempty(sel.tr_noise_rate) || ~isempty(sel.tr_occlusion_rate)) && ...
                                isempty(tci_offsets))
                                tci_offsets = find(sel.tr_col_indices == col_idx);
                            end

                            % Check whether col_idx is a noise required index 
                            noise_rate = [];
                            if (~isempty(sel.tr_noise_rate) && ...
                                    (tci_offsets(dsi) <= numel(sel.tr_noise_rate)) && ...
                                    (0 ~= sel.tr_noise_rate(tci_offsets(dsi))))
                                noise_rate = sel.tr_noise_rate(tci_offsets(dsi));                                
                            end

                            % Check whether col_idx is a occlusion required index 
                            occl_rate = [];
                            occl_type = [];
                            occl_offset = 0;
                            if (~isempty(sel.tr_occlusion_rate) && ...
                                    (tci_offsets(dsi) <= numel(sel.tr_occlusion_rate)) && ...
                                    (0 ~= sel.tr_occlusion_rate(tci_offsets(dsi))))
                                occl_rate = sel.tr_occlusion_rate(tci_offsets(dsi));

                                if (~isempty(sel.tr_occlusion_type))
                                    occl_type = sel.tr_occlusion_type(tci_offsets(dsi));
                                end
                                
                                if (~isempty(sel.tr_occlusion_offset))
                                    occl_offset = sel.tr_occlusion_offset(tci_offsets(dsi));
                                end
                            end
                            
                            DbSlice.build_nndb_tr_(nndbs, pi, is_new_class, pimg, ...
                                                          noise_rate, occl_rate, occl_type, occl_offset);                            

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
                nndbs_tr = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TR));
                nndbs_val = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.VAL));
                nndbs_te = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TE));
                nndbs_tr_out = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TR_OUT));
                nndbs_val_out = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.VAL_OUT));
                nndbs_te_out = DbSlice.p0_nndbs_(dict_nndbs, uint32(Dataset.TE_OUT));
                edatasets = [Dataset.TR Dataset.VAL Dataset.TE Dataset.TR_OUT Dataset.VAL_OUT Dataset.TE_OUT];
                return;
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
            % Peform image preocessing.
        
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
        function examples(imdb_8)
            %EXAMPLES: Extensive example set
            %   Assume:
            %   There are only 8 images per subject in the database. 
            %   NNdb is in H x W x CH x N format. (image database)
            %
            
            %%% Full set of options
            % nndb = NNdb('original', imdb_8, 8, true);
            % sel.tr_col_indices        = [1:3 7:8]; %[1 2 3 7 8]; 
            % sel.tr_noise_rate         = [];
            % sel.tr_occlusion_rate     = [];
            % sel.tr_occlusion_type     = [];
            % sel.tr_occlusion_offset   = [];
            % sel.tr_out_col_indices    = [];
            % sel.val_col_indices       = [];
            % sel.val_out_col_indices   = [];
            % sel.te_col_indices        = [4:6]; %[4 5 6]
            % sel.te_out_col_indices    = [];
            % sel.nnpatches             = [];
            % sel.use_rgb               = false;
            % sel.color_index           = [];                
            % sel.use_real              = false;
            % sel.scale                 = 0.5;
            % sel.normalize             = false;
            % sel.histeq                = true;
            % sel.histmatch_col_index   = [];
            % sel.class_range           = [1:36 61:76 78:100];
            % sel.val_class_range       = [];
            % sel.te_class_range        = [];
            % %sel.pre_process_script   = @fn_custom_pprocess;
            % sel.pre_process_script    = [];
            % [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel); 
            
            % Imports
            import nnf.db.NNdb;
            import nnf.db.DbSlice;
            import nnf.db.Selection;
            import nnf.core.generators.NNPatchGenerator;
            import nnf.db.Noise;
                        
            % 
            % Select 1st 2nd 4th images of each identity for training.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4]; %[1 2 4]; 
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel); % nndb_tr = DbSlice.slice(nndb, sel); 

            
            %
            % Select 1st 2nd 4th images of each identity for training.
            % Divide into patches
            nndb = NNdb('original', imdb_8, 8, true);
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
            % Select 1st 2nd 4th images of each identity for training.
            % Select 3rd 5th images of each identity for testing.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
           	% 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 3rd 5th images of identities denoted by class_range for testing.            
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            %
            % Select 1st and 2nd image from 1st class and 2nd, 3rd and 5th image from 2nd class for training 
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection()
            sel.tr_col_indices      = {[1 2], [2 3 5]};
            sel.class_range         = [1:2];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 1st 2nd 4th images images of identities denoted by class_range for validation.   
            % Select 3rd 5th images of identities denoted by class_range for testing. 
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.val_col_indices= [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            [nndb_tr, nndb_val, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 1st 2nd 4th images images of identities denoted by val_class_range for validation.   
            % Select 3rd 5th images of identities denoted by te_class_range for testing. \
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.val_col_indices= [1:2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.class_range    = [1:10];    % First ten identities 
            sel.val_class_range= [6:15];
            sel.te_class_range = [17:20];
            [nndb_tr, nndb_val, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of identities denoted by class_range for training.
            % Select 3rd 4th images of identities denoted by val_class_range for validation.
            % Select 3rd 5th images of identities denoted by te_class_range for testing.
            % Select 1st 1st 1st images of identities denoted by class_range for training target.
            % Select 1st 1st images of identities denoted by val_class_range for validation target.
            nndb = NNdb('original', imdb_8, 8, true);
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
            % Training column will consists of first 60% of total columns avaiable
            % Testing column will consists of first 40% of total columns avaiable
            nndb = NNdb('original', imdb_8, 8, True);
            sel = Selection();
            sel.tr_col_indices      = Select.PERCENT_60;
            sel.te_col_indices      = Select.PERCENT_40;
            sel.class_range         = [1:10];
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] =...
                                    DbSlice.slice(nndb, sel);
            nndb_tr.show(10, 5);
            nndb_te.show(10, 4);


            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add various noise types @ random locations of varying degree.
            %               default noise type: random black and white dots.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];   %[1 2 4]; 
            sel.tr_noise_rate  = [0 0.5 0.2];       % percentage of corruption
            %sel.tr_noise_rate  = [0 0.5 Noise.G];  % last index with Gauss noise
            sel.class_range    = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);

            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add various occlusion types ('t':top, 'b':bottom, 'l':left, 'r':right, 'h':horizontal, 'v':vertical) of varying degree.
            %               default occlusion type: 'b'.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];
            sel.tr_occlusion_rate = [0 0.5 0.2];    % percentage of occlusion
            sel.tr_occlusion_type = 'ttt';          % occlusion type: 't' for selected tr. indices [1, 2, 4]
            %sel.tr_occlusion_type = 'tbr' 
            %sel.tr_occlusion_type = 'lrb' 
            sel.class_range = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            
            
            % 
            % Select 1st 2nd 4th images of each identity for training + 
            %               add occlusions in the middle (horizontal/vertical).
            %               default occlusion type: 'b'.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1:2 4];
            sel.tr_occlusion_rate = [0 0.5 0.2];        % percentage of occlusion
            sel.tr_occlusion_type = 'ttt';              % occlusion type: 't' for selected tr. indices [1, 2, 4]
            sel.tr_occlusion_offset = [0 0.25 0.4];     % Offset from top since 'tr_occlusion_type = t'            
            % sel.tr_occlusion_type = 'rrr';            % occlusion type: 'r' for selected tr. indices [1, 2, 4]
            % sel.tr_occlusion_offset = [0 0.25 0.4];   % Offset from right since 'tr_occlusion_type = r'
            sel.class_range = [1:10];
            [nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
            

           	% 
            % To prepare regression datasets, training dataset and training target dataset
            % Select 1st 2nd 4th images of each identity for training.
            % Select 1st 1st 1st image of each identity for corresponding training target.
            % Select 3rd 5th images of each identity for testing.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];       %[1 2 4]; 
            sel.tr_out_col_indices = [1 1 1];   %[1 1 1]; (Regression 1->1, 2->1, 4->1) 
            sel.te_col_indices = [3 5];         %[3 5]; 
            [nndb_tr, ~, nndb_te, nndb_tr_out, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %  
            % Resize images by 0.5 scale factor.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.scale          = 0.5;
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %  
            % Use gray scale images.
            % Perform histrogram equalization.
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.histeq         = true;
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %  
            % Use gray scale images.
            % Perform histrogram match. This will be performed with the 1st image of each identity
            % irrespective of the selection choice. (refer code for more details)
            nndb = NNdb('original', imdb_8, 8, true);
            sel = Selection();
            sel.tr_col_indices = [1 2 4];   %[1 2 4]; 
            sel.te_col_indices = [3 5];     %[3 5]; 
            sel.use_rgb        = false;
            sel.histmatch_col_index = 1;
            [nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);


            %
            % If imdb_8 supports many color channels
            nndb = NNdb('original', imdb_8, 8, true);
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
        function [filter] = get_occlusion_patch(h, w, dtype, occl_type, occl_rate, occl_offset)
            
            % Set defaults for arguments
            if (nargin < 6); occl_offset = 0; end
            
            filter = ones(h, w);
            filter = cast(filter, dtype);
            
            if (isempty(occl_type) || occl_type == 'b')
                sh = ceil(occl_rate * h);
                en = floor((1-occl_offset) * h); 
                st = en - sh + 1; if (st < 0); st = 1; end
                filter(st:en, 1:w) = 0;

            elseif ((occl_type == 'r'))
                sh = ceil(occl_rate * w);
                en = floor((1-occl_offset) * w); 
                st = en - sh + 1; if (st < 0); st = 1; end
                filter(1:h, st:en) = 0;

            elseif (occl_type == 't' || occl_type == 'v')
                sh = floor(occl_rate * h);
                st = floor(occl_offset * h) + 1; 
                en = st + sh - 1; if (en > h); en = h; end
                filter(st:en, 1:w) = 0;

            elseif (occl_type == 'l' || occl_type == 'h')
                sh = floor(occl_rate * w);
                st = floor(occl_offset * w) + 1; 
                en = st + sh - 1; if (en > w); en = w; end
                filter(1:h, st:en) = 0;              
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
            % col_ranges : describe
            %     decribe.
            % 
            % Returns
            % -------
            % dict_nndbs : describe
            %

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
        function nndbs = build_nndb_tr_(nndbs, pi, is_new_class, img, ...
                                                        noise_rate, occl_rate, occl_type, occl_offset) 
            % BUILD_NNDB_TR: builds the nndb training database.
            %          
            % Returns
            % -------
            % nndbs : `array_like` -nnf.db.NNdb
            %      Vector of patch databases.
            %  
            
            % Imports 
            import nnf.db.DbSlice;
            import nnf.db.Format;
            import nnf.db.Noise;
             
            if (isempty(nndbs)); return; end
            nndb = nndbs(pi);
            
            if (~isempty(occl_rate) || ~isempty(noise_rate))
                
                [h, w, ch] = size(img);
                
                % Adding different occlusions depending on the precentage
                if (~isempty(occl_rate))
                    filter = DbSlice.get_occlusion_patch(h, w, class(img), occl_type, occl_rate, occl_offset);                    
                    
                    % For grey scale
                    if (ch == 1)
                        img = filter.*img;
                    else
                        % For colored
                        img = repmat(filter, 1, 1, ch).*img;
                    end
                    
                % Add different noise depending on the type or rate
                % (ref. Enums/Noise)
                elseif (~isempty(noise_rate) && (noise_rate == Noise.G))
                    img = imnoise(img, 'gaussian');
                    % img = imnoise(img, 'gaussian');
                    % img = imnoise(img, 'gaussian');
                    % img = imnoise(img, 'gaussian');
                    
                elseif (~isempty(noise_rate))
                    % Perform random corruption
                    % Corruption Size (H x W)
                    cs = [uint16(h*noise_rate) uint16(w*noise_rate)]; 

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
            
            % Add data to nndb
            nndb.add_data(img);

            % Update the properties of nndb
            nndb.update_attr(is_new_class);        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function img = rand_corrupt_(height, width) 
            % RAND_CORRUPT: corrupts the image with a (height, width) block.
            %            
            % Returns
            % -------
            % img : `array_like` -uint8
            %      3D tensor to representing corrupted image.
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
        function nndbs = build_nndb_tr_out_(nndbs, pi, is_new_class, img)
            % BUILD_NNDB_TR_OUT: builds the nndb training target database.
            %            
            % Returns
            % -------
            % nndbs : `array_like` -nnf.db.NNdb
            %      Vector of patch databases.
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
            % Returns
            % -------
            % nndbs : `array_like` -nnf.db.NNdb
            %      Vector of patch databases.
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
            % Returns
            % -------
            % nndbs : `array_like` -nnf.db.NNdb
            %      Vector of patch databases.
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
            % Returns
            % -------
            % nndbs : `array_like` -nnf.db.NNdb
            %      Vector of patch databases.
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
            % Returns
            % -------
            % nndbs : `array_like` -nnf.db.NNdb
            %      Vector of patch databases.
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
    end
end