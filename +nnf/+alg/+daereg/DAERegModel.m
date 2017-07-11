classdef (Abstract) DAERegModel < handle
    %DAEReg: Deep Autoencoder Regression Model.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties (SetAccess = public)
        name;             % (s) Name of the instance
        tr_indices;       % (v) Training indices
        val_indices;      % (v) Validation indices
        te_indices;       % (v) Validation indices
        
        nndbs;            % Cell array of `NNdb` used for layer-wise pretraining
        split_val_data;   % Split the dataset into a validation dataset
        fine_tune;        % Whether to fine-tune or not
        %dict_nndb_ref;
        
        in_idx;           % Input database index
        out_idx;          % Output database index
    end
    
    properties (SetAccess = protected)
       user_data_; 
    end
    
    methods(Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nncfg = get_sdae_nncfg(dr_arch, rl_arch)
            % Get sparse deep autoencoder regression neural network configuration.
            %
            % Parameters
            % ----------
            % dr_arch : vector -double
            %   Dimension reduction net architecture.
            %
            % rl_arch : vector -double
            %   Regression net architecture.
            %
            % Returns
            % -------
            % nncfg : struct
            %   nncfg.aes = array of AECfg objects (dimension reduction nets)
            %   nncfg.nets = array of AECfg objects (regression nets)
            %
            %   % Fine-tune related for complete network
            %   nncfg.cost_fn
            %   nncfg.reg_ratio
            %

            % Imports
            import nnf.alg.daereg.AECfg;
            
            % Pre-Training
            % Layers before last layer (pre-trained with AEs)
            nncfg.aes = [];
            for i=1:numel(dr_arch)
                ae = AECfg(dr_arch(i));
                nncfg.aes = [nncfg.aes ae];
            end
            
            % Fitting nets
            nncfg.nets = [];
            for i=1:numel(rl_arch)
                rlnet = AECfg(rl_arch(i));
                nncfg.nets = [nncfg.nets rlnet];            
            end
                        
            % Fine Tuning
            % Complete network
            nncfg.reg_ratio = 0.0001;
            nncfg.cost_fn   = 'mse';            
            nncfg.trainFcn            = 'trainscg';
            nncfg.trainParam.goal     = 1e-10;
            nncfg.trainParam.sigma    = 1e-11;
            nncfg.trainParam.lambda   = 1e-9;
            nncfg.trainParam.epochs   = 125000;
            nncfg.trainParam.max_fail = 4100;
            nncfg.trainParam.min_grad = 1e-10; 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nncfg = get_dae_info(dr_arch, rl_arch, actfn)
            % Get deep autoencoder regression neural network configuration.
            %
            % Parameters
            % ----------
            % dr_arch : vector -double
            %   Dimension reduction net architecture.
            %
            % rl_arch : vector -double
            %   Regression net architecture.
            %
            % Returns
            % -------
            % nncfg : struct
            %   nncfg.aes = array of AECfg objects (dimension reduction nets)
            %   nncfg.nets = array of AECfg objects (regression nets)
            %
            %   % Fine-tune related for complete network
            %   nncfg.cost_fn
            %   nncfg.reg_ratio
            %
            
            % Imports
            import nnf.alg.daereg.AECfg;
            
            if (nargin < 4) 
                actfn = 'logsig';
            end
                
            % Pre-Training
            % Layers before last layer (pre-trained with AEs)
            nncfg.aes = [];
            for i=1:numel(dr_arch)
                aecfg = AECfg(dr_arch(i));
                aecfg.enc_fn = actfn; 
                aecfg.cost_fn = 'mse';                
                aecfg.sparse_reg = 0;
                aecfg.sparsity = 0;
                aecfg.l2_wd = 0;
                aecfg.reg_ratio = 0;                
                nncfg.aes = [nncfg.aes aecfg];
            end
            
            % Fitting nets
            nncfg.nets = [];
            for i=1:numel(rl_arch)
                % Fitting net
                rlnet = AECfg(rl_arch(i)); 
                rlnet.enc_fn = 'tansig'; 
                rlnet.cost_fn = 'mse';
                aecfg.sparse_reg = 0;
                aecfg.sparsity = 0;
                aecfg.l2_wd = 0;
                rlnet.reg_ratio = 0;                
                nncfg.nets = [nncfg.nets rlnet];            
            end
            
            % Fine Tuning
            % Complete network
            nncfg.reg_ratio = 0.0001;
            nncfg.cost_fn   = 'mse';
            nncfg.trainFcn            = 'trainscg';
            nncfg.trainParam.goal     = 1e-10;
            nncfg.trainParam.sigma    = 1e-6;
            nncfg.trainParam.lambda   = 1e-5;
            nncfg.trainParam.epochs   = 125000;
            nncfg.trainParam.max_fail = 4100;
            nncfg.trainParam.min_grad = 1e-10;

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nncfg = get_basic_dae_info(enc_arch, act_fns)
            % Get basic deep autoencoder neural network configuration.
            %
            % Parameters
            % ----------
            % enc_arch : vector -double
            %   Encoding architecture for the basic DAE.
            %
            % act_fns : cell -string
            %   Activation functions for the hidden layer architecture.
            %
            % Returns
            % -------
            % nncfg : struct
            %   nncfg.aes = array of AECfg objects (dimension reduction nets)
            %
            %   % Fine-tune related for complete network
            %   nncfg.cost_fn
            %   nncfg.reg_ratio
            %
            
            % Imports
            import nnf.alg.daereg.AECfg;
            
            % Pre-Training
            % Layers before last layer (pre-trained with AEs)
            nncfg.aes = [];
            for i=1:numel(enc_arch) % enc_arch = [50 30]
                aecfg = AECfg(enc_arch(i));
                aecfg.enc_fn = act_fns{i}; 
                aecfg.cost_fn = 'mse';                
                aecfg.sparse_reg = 0;
                aecfg.sparsity = 0;
                aecfg.l2_wd = 0;
                aecfg.reg_ratio = 0;                
                nncfg.aes = [nncfg.aes aecfg];
            end
            
            % Fine Tuning
            % Complete network
            nncfg.arch = [enc_arch fliplr(enc_arch(1:end-1))];
            nncfg.reg_ratio = 0.0001;
            nncfg.reg_ratio = 0;
            nncfg.cost_fn   = 'mse';
            nncfg.trainFcn            = 'trainscg';
            nncfg.trainParam.goal     = 1e-10;
            nncfg.trainParam.sigma    = 1e-6;
            nncfg.trainParam.lambda   = 1e-5;
            nncfg.trainParam.epochs   = 125000;
            nncfg.trainParam.max_fail = 4100;
            nncfg.trainParam.min_grad = 1e-10;            

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nncfg = get_dnn_info(arch, use_scg)
            % Get deep neural network configuration.
            %
            % Parameters
            % ----------
            % arch : vector -double
            %   net architecture.
            %
            % Returns
            % -------
            % nncfg : struct
            %   nncfg.dnn = deep neural network configuration
            %

            if (nargin < 3); use_scg = true; end
            
            % DNN configuration
            nncfg.dnn.arch = arch;  
            
            % Fine Tuning
            % Complete network
            nncfg.reg_ratio = 0;
            nncfg.cost_fn   = 'mse';
                
            if (use_scg)
                nncfg.trainFcn            = 'trainscg';
                nncfg.trainParam.goal     = 1e-10;
                nncfg.trainParam.sigma    = 1e-11;
                nncfg.trainParam.lambda   = 1e-9;
                nncfg.trainParam.epochs   = 125000;
                nncfg.trainParam.max_fail = 4100;
                nncfg.trainParam.min_grad = 1e-10;                
            else                
                nncfg.trainFcn            = 'traingdx';
                nncfg.trainParam.goal     = 1e-10;
                nncfg.trainParam.lr       = 100;
                nncfg.trainParam.mc       = 0.9;
                nncfg.trainParam.epochs   = 1000000;
                nncfg.trainParam.max_fail = 4100;
                nncfg.trainParam.min_grad = 1e-10;                
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
        
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface       
       	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = DAERegModel(name, nndbs, split_val_data, fine_tune)
            % Constructs a DAERegModel object.
            %
            % Parameters
            % ----------
            % name : str
            %     Name of the instance.
            %
            % nndbs : cell -NNdb
            %     Cell array of `NNdb` used for layer-wise pretraining.
            %
            % split_val_data : bool
            %     Split the dataset into a validation dataset.
            %
            disp(['Costructor::DAERegModel ' name]);
            
            if (nargin < 3)
                split_val_data = false;
            end
            if (nargin < 4)
                fine_tune = true;
            end
            
            self.name = name;
            self.nndbs = nndbs;
            self.in_idx = 1;
            self.out_idx = 2;
            self.split_val_data = split_val_data;
            self.fine_tune = fine_tune;
                        
            % For all follow up random operations, initialize the same seed
            rng('default');
            
            % Init
            self.init(nndbs);
        end        
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function init(self, nndbs, indices)
            % Initialize `DAERegModel` instance.
            % 
            % Parameters
            % ----------
            % nndbs : cell -NNdb
            %     Cell array of `NNdb` used for layer-wise pretraining.
            %            
            
            % If no explicit indices were givens
            if (nargin < 3)            
                n = nndbs{self.in_idx}.n;

                % Main division
                if (self.split_val_data)
                    [self.tr_indices, self.val_indices, self.te_indices] = dividerand(n, 70/100, 15/100, 15/100);
                else
                    [self.tr_indices, self.val_indices, self.te_indices] = dividerand(n, 85/100, 0, 15/100);
                end
                
            else  
                % Load explicit indices
                self.tr_indices = indices{1};
                self.val_indices = indices{2};
                self.te_indices = indices{3};
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function indices = get_indices(self)
            indices = {self.tr_indices, self.val_indices, self.te_indices};
        end
 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Test cases               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function TC_base(self, nncfg, pp_infos)
            % Base test case.
            % 
            % Parameters
            % ----------
            % nndbs : cell -NNdb
            %     Cell array of `NNdb` used for layer-wise pretraining.
            %
            % nncfg : struct
            %     Neural network configuration structure.
            %
            %       % Pre-Training related
            %       nncfg.aes = array of AECfg objects (dimension reduction nets)
            %       nncfg.nets = array of AECfg objects (regression nets)
            %
            %       % Fine-tune related
            %       nncfg.cost_fn
            %       nncfg.reg_ratio
            %
            % pp_infos : cell -struct
            %     Cell array of `struct` that indicates the pre-processing required.
            %       For ith nndb the pp_info is shown below.
            %           pp_infos{i}.removeconstantrows = false
            %           pp_infos{i}.diff = false
            %           pp_infos{i}.mapstd = false
            %           pp_infos{i}.whiten = false
            %           pp_infos{i}.mapminmax  % definition of the field
            %

            pp_infos = self.pre_process(pp_infos);          
            self.train_and_eval(nncfg, pp_infos);
        end
                       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pp_infos = pre_process(self, pp_infos)
            % Pre-process the nndbs.
            % 
            % Parameters
            % ----------
            % pp_infos : cell -struct
            %     Cell array of `struct` that indicates the pre-processing required.
            %       For ith nndb the pp_info is shown below.
            %           pp_infos{i}.removeconstantrows = false
            %           pp_infos{i}.diff = false
            %           pp_infos{i}.mapstd = false
            %           pp_infos{i}.whiten = false
            %           pp_infos{i}.mapminmax  % definition of the field
            %
            
            % Imports
            import nnf.utl.*;
            import nnf.db.Format;
            import nnf.db.NNdb;
                        
            % Set defaults for arguments
            if (isempty(pp_infos))
                pp_infos = cell(1, numel(self.nndbs)); 
            end
            
            % Set defaults for info fields, if the field does not exist
            for i=1:numel(self.nndbs)
                if (i > numel(pp_infos) || isempty(pp_infos{i}))
                     pp_infos{i} = struct;
                end
            end
                        
            % %TODO: Issue user warning if pre-processing is required in layers other than 1
            % pp_1_st_layer = true;
            % pp_fcns = {'removeconstantrows', 'diff', 'mapstd'};
            % for i=1:numel(pp_infos)
            % 
            %     for j=1:numel(pp_fcns)
            % 
            %         eval(sprintf('b = pp_infos{%d}.%s', i, pp_fcns{j}))
            % 
            %         if ((i==1) && (~isfield(pp_infos{i}, pp_fcns{j}) || ~b))
            %             pp_1_st_layer = false;
            % 
            %         elseif ((i~=1) && (isfield(pp_infos{i}, 'removeconstantrows') && pp_infos{i}.removeconstantrows))
            %             if (~pp_1_st_layer)
            %                 warn(['Preprocessing for layer (' num2str(i) ...
            %                     ') will be ignored since the 1st layer preprocessing is not requested.'])
            %             end
            %         end
            %     end
            % end
            
            % PERFORM REMOVECONSTANTROWS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Note: Should be done explicitly since no support in GPU network training
            for i=1:numel(self.nndbs)
                pp_info = pp_infos{i}; 

                if (~isfield(pp_info, 'removeconstantrows') || ~pp_info.removeconstantrows)
                    continue;
                end
                nndb = self.nndbs{i};

                % Remove constant rows
                db = removeconstantrows(nndb.features);                        
                self.nndbs{i} = NNdb(['db' num2str(i) '_removeconstantrows'], db, nndb.n_per_class, false, nndb.cls_lbl, Format.H_N);                    
            end
            
            % PERFORM DIFF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i=1:numel(self.nndbs)
                pp_info = pp_infos{i}; 

                if (~isfield(pp_info, 'diff') || ~pp_info.diff)
                    continue;
                end
                nndb = self.nndbs{i};

                % Differences from the base
                format = nndb.format;
                if (format == Format.H_W_CH_N)
                    base = double(nndb.db(:, :, :, 1));
                    db = bsxfun(@minus, base, nndb);                        
                else
                    base = nndb.features(:, 1);
                    db = bsxfun(@minus, base, nndb.features);
                end

                self.nndbs{i} = NNdb('db_diff', db, nndb.n_per_class, false, nndb.cls_lbl, format);                    
            end
            
            % PERFORM MAPSTD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i=1:numel(self.nndbs)
                pp_info = pp_infos{i}; 

                if (~isfield(pp_info, 'mapstd') || ~pp_info.mapstd)
                    continue;
                end
                nndb = self.nndbs{i};

                % Map std
                db = mapstd(nndb.features);
                self.nndbs{i} = NNdb(['db' num2str(i) '_std'], db, nndb.n_per_class, false, nndb.cls_lbl, Format.H_N);
            end
            
            % PERFORM WHITENING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
            % Perform whitening for the nndbs
            for i=1:numel(self.nndbs)
                pp_info = pp_infos{i};                  

                % First nndb
                if (isfield(pp_info, 'whiten') && (isfield(pp_info.whiten, 'run') && pp_info.whiten.run))
                    nndb = self.nndbs{i};                        

                    keep_dims = (isfield(pp_info.whiten, 'keep_dims') && pp_info.whiten.keep_dims);
                    squash_to_zero = (isfield(pp_info.whiten, 'squash_to_zero') && pp_info.whiten.squash_to_zero);

                    tr_db = self.nndbs{i}.features(:, self.tr_indices);
                    val_db = self.nndbs{i}.features(:, self.val_indices);
                    te_db = self.nndbs{i}.features(:, self.te_indices);

                    % Learn whiten projection from tr_db
                    [tr_db, W, r_info] = whiten(tr_db, 1e-5, keep_dims, squash_to_zero);

                    % Apply it to val_db and te_db
                    val_db = W'* bsxfun(@minus, val_db, r_info.m);
                    te_db  = W'* bsxfun(@minus, te_db, r_info.m);

                    db = zeros(size(tr_db, 1), nndb.n);
                    db(:, self.tr_indices) = tr_db;
                    db(:, self.val_indices) = val_db;
                    db(:, self.te_indices) = te_db;

                    self.nndbs{i} = NNdb(['db' num2str(i) '_whiten'], db, nndb.n_per_class, false, nndb.cls_lbl, Format.H_N);

                    % Clear variables
                    clear tr_db;
                    clear te_db;

                    % Store whitening info
                    pp_infos{i}.int.whiten.W = W;
                    pp_infos{i}.int.whiten.r_info = r_info;

                else
                    % Rest of the nndbs
                    if (~isfield(pp_info, 'whiten') || ~isfield(pp_info.whiten, 'apply'))
                        continue;
                    end

                    nndbs_idx = pp_info.whiten.apply;
                    W = pp_infos{nndbs_idx}.int.whiten.W;
                    r_info = pp_infos{nndbs_idx}.int.whiten.r_info;                        

                    nndb = self.nndbs{i};
                    db = W' *  bsxfun(@minus, nndb.features, r_info.m);
                    self.nndbs{i} = NNdb(['db' num2str(i) '_whiten'], db, nndb.n_per_class, false, nndb.cls_lbl, Format.H_N);

                    % Store whitening info
                    pp_infos{i}.int.whiten.W = W;
                    pp_infos{i}.int.whiten.r_info = r_info;
                end
            end

            % Clear variables
            clear db;
            clear W;
            clear m;
            
            % PERFORM MAPMINMAX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % For later reference
            % Find the string 'mapminmax' in first autoencoder and perform mapminmax pre-processing
            % if (~isempty(find(not(cellfun('isempty', strfind(nncfg.aes(1).inProcessFcns, 'mapminmax'))))))            
            pp_infos = pp_map_min_max_(self, pp_infos);
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function train_and_eval(self, nncfg, pp_infos)
            % Train and evaluate the `DAERegModel`.
            % 
            % Parameters
            % ----------
            % nndbs : cell -NNdb
            %     Cell array of `NNdb` used for layer-wise pretraining.
            %
            % info : struct
            %     Information structure.
            %
            % Notes
            % -----
            % Following fields of `info` are used.
            %
            %   % Pre-Training related
            %   nncfg.aes = array of AECfg objects (dimension reduction nets)
            %   nncfg.nets = array of AECfg objects (regression nets)
            %
            %   % Fine-tune related
            %   nncfg.cost_fn
            %   nncfg.reg_ratio
            %

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Pre-training layers with Autoencoders (Dimensionality Reduction)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % defaults: 
            %   divideFcn:  dividetrain
            %   trainFcn:   trainscg
            %   performFcn: msesparse
            %       
            
            % Callback
            self.on_train_start_(pp_infos);
            
            if (self.split_val_data)
                XX = self.nndbs{self.in_idx}.features;
            else                
                XX = self.nndbs{self.in_idx}.features(:, self.tr_indices);
            end
            XX_te = self.nndbs{self.in_idx}.features(:, self.te_indices);
			XX_val = self.nndbs{self.in_idx}.features(:, self.val_indices);

            % Configure the network count
            aes_n = 0;
            nets_n = 0;
            if (isfield(nncfg, 'aes'))
                aes_n = numel(nncfg.aes);
            end            
            if (isfield(nncfg, 'nets'))
                nets_n = numel(nncfg.nets);
            end
            
            % Initialize the pretr_nets and tr_stats cell arrays to store to training related info
            net_idx = 1;
            pretr_nets = cell(1, aes_n + nets_n);
            tr_stats = cell(1, aes_n + nets_n + 1);            
            total_time = 0;
            
            loaded = []; deepnet = [];
            if (isfield(nncfg, 'load_from')); loaded = load(nncfg.load_from); end
            if (isfield(nncfg, 'load_deepnet') && nncfg.load_deepnet)
                if (isempty(loaded)); error("Please specify 'nncfg.load_from'"); end                
                %pretr_nets = loaded.pretr_nets;
                tr_stats = loaded.tr_stats;
                deepnet = loaded.deepnet;
            end

            % Only if deepnet is not already loaded
            if (isempty(deepnet)) 
                for i=1:aes_n
                    aecfg = nncfg.aes(i);
                    aecfg.validate();

                    if (isfield(nncfg, 'load_aes') && nncfg.load_aes(i))
                        if (isempty(loaded)); error("Please specify 'nncfg.load_from'"); end
                        autoenc = loaded.pretr_nets{net_idx};
                        tr_stat = loaded.tr_stats{net_idx};

                    else
                        % trainAutoencoder only supports 'msesparse' by default. Thus customized.
                        [autoenc, tr_stat] = self.trainAutoencoder(XX, XX, aecfg, aecfg.hidden_nodes, ...
                            'ScaleData', false,...
                            'useGPU', true);
                            %'EncoderTransferFunction',aecfg.enc_fn,...
                            %'DecoderTransferFunction',aecfg.dec_fn,...
                            %'L2WeightRegularization',aecfg.l2_wd,...
                            %'SparsityRegularization',aecfg.sparse_reg,...
                            %'SparsityProportion',aecfg.sparsity,...               
                    end

                    total_time = total_time + sum(tr_stat.time);
                    pretr_nets{net_idx} = autoenc;
                    tr_stats{net_idx} = tr_stat;
                    net_idx = net_idx + 1;

                    % Display Performance For Autoencoder i
                    PXX         = predict(autoenc, XX);
                    mse_err_tr  = mse(XX - PXX);
                    r_tr        = regression(XX, PXX, 'one'); 
                    PXX_te      = predict(autoenc, XX_te);
                    mse_err_te  = mse(XX_te - PXX_te);
                    r_te        = regression(XX_te, PXX_te, 'one');
                    e_te        = gsubtract(XX_te, PXX_te);
                    PXX_val     = predict(autoenc, XX_val);
                    mse_err_val = mse(XX_val - PXX_val);
                    r_val       = regression(XX_val, PXX_val, 'one');                
                    e_val       = gsubtract(XX_val, PXX_val);

                    disp(['AE' num2str(i) '_ERR: Tr:' num2str(mse_err_tr) ' R:' num2str(r_tr) ...
                            ' Te:' num2str(mse_err_te) ' R:' num2str(r_te) ' S:(m:' num2str(mean(e_te(:))) ', d:' num2str(std(e_te(:))) ')'...
                            ' Val:' num2str(mse_err_val) ' R:' num2str(r_val) ' S:(m:' num2str(mean(e_val(:))) ', d:' num2str(std(e_val(:))) ')'...
                            ' Time:' num2str(total_time/1000)]);

                    % Encode the information for the next round
                    XX     = encode(autoenc, XX);
                    XX_te  = encode(autoenc, XX_te);
                    XX_val  = encode(autoenc, XX_val);
                end 
            end
                
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Pre-training layers with Autoencoders (Relationship Mapping)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
            XX = self.on_dr_end_(pretr_nets(1:aes_n), XX); % Callback
            XX_val = XX(:, self.val_indices);
            
            % Only if deepnet is not already loaded
            if (isempty(deepnet))
                for i=1:nets_n
                    netcfg = nncfg.nets(i);
                    [XXT, XXT_te, XXT_val] = self.on_rl_loop_init_(i);

                    % Normalizing output for relationship learning AE.
                    % netcfg.outProcessFcns{1} = 'mapminmax';

                    if (isfield(nncfg, 'load_nets') && nncfg.load_nets(i))                        
                       if (isempty(loaded)); error("Please specify 'nncfg.load_from'"); end
                       autoenc = loaded.pretr_nets{net_idx};
                       tr_stat = loaded.tr_stats{net_idx};

                    else
                        % Train the Network (GPU)
                        [autoenc, tr_stat] = self.trainAutoencoder(XX, XXT, netcfg, netcfg.hidden_nodes, ...
                                    'ScaleData', false,...
                                    'useGPU', true);
                    end

                    total_time = total_time + sum(tr_stat.time);
                    pretr_nets{net_idx} = autoenc;
                    tr_stats{net_idx} = tr_stat;
                    net_idx = net_idx + 1;

                    % Display Performance For Autoencoder i  
                    PXX         = predict(autoenc, XX);
                    mse_err_tr  = mse(XXT - PXX);
                    r_tr        = regression(XXT, PXX, 'one'); 
                    PXX_te      = predict(autoenc, XX_te);
                    mse_err_te  = mse(XXT_te - PXX_te);
                    r_te        = regression(XXT_te, PXX_te, 'one');	
                    e_te        = gsubtract(XXT_te, PXX_te);
                    PXX_val     = predict(autoenc, XX_val);
                    mse_err_val = mse(XXT_val - PXX_val);
                    r_val       = regression(XXT_val, PXX_val, 'one');
                    e_val       = gsubtract(XXT_val, PXX_val);

                    disp(['NT' num2str(i) '_ERR: Tr:' num2str(mse_err_tr) ' R:' num2str(r_tr) ...
                            ' Te:' num2str(mse_err_te) ' R:' num2str(r_te) ' S:(m:' num2str(mean(e_te(:))) ', d:' num2str(std(e_te(:))) ')'...
                            ' Val:' num2str(mse_err_val) ' R:' num2str(r_val) ' S:(m:' num2str(mean(e_val(:))) ', d:' num2str(std(e_val(:))) ')'...
                            ' Time:' num2str(total_time/1000)]);

                    % Encode the information for the next round
                    XX     = encode(autoenc, XX);
                    XX_te  = encode(autoenc, XX_te);
                    XX_val = encode(autoenc, XX_val);
                end
            end

            % Sample Code to extract features from `net`
            % XXF = tansig(net.IW{1} * XXF + repmat(net.b{1}, 1, size(XXF, 2)));
            % XXF_te = tansig(net.IW{1} * XXF_te + repmat(net.b{1}, 1, size(XXF_te, 2)));  

            % Setup data for fine tuning
            if (self.split_val_data)
                XX = self.nndbs{self.in_idx}.features;
                YY = self.nndbs{self.out_idx}.features;
            else                
                XX = self.nndbs{self.in_idx}.features(:, [self.tr_indices self.val_indices]);
                YY = self.nndbs{self.out_idx}.features(:, [self.tr_indices self.val_indices]);
            end
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stack the layers for a Deep Network and Fine-Tune
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Only if deepnet is not already loaded
            if (isempty(deepnet))
                
                % If `regression` nets are present
                if (nets_n > 0)
                    if (numel(pretr_nets) >= 2)
                        deepnet = pretr_nets{1};
                        for i=2:numel(pretr_nets)-1
                            deepnet = stack(deepnet, pretr_nets{i}); 
                        end    

                        % The encoder/decoder weights and the output layer for deep net is configured when 'net' is
                        % used with stack(...). When 'Autoencoder' is used with stack(...), the encoder weights
                        % along with the encoder will be used but not decoding segment.
                        deepnet = stack(deepnet, pretr_nets{end}.network);
                    else
                        deepnet = pretr_nets{1}.network;
                    end

                % Deep neural network
                elseif (isfield(nncfg, 'dnn'))
                    % Building model from the scratch, uncomment if necessary
                    % layer_n = numel(nncfg.dnn.arch) + 1; % hidden layers + output
                    % biasConnect = ones(layer_n, 1);                
                    % inputConnect = zeros(layer_n, 1);
                    % inputConnect(1) = 1;
                    % layerConnect = ones(layer_n - 1, 1);
                    % layerConnect = diag(layerConnect, -1);
                    % outputConnect = zeros(1, layer_n);
                    % outputConnect(end) = 1;
                    % deepnet = network(1, layer_n, ...
                    %                     logical(biasConnect), ...
                    %                     logical(inputConnect), ...
                    %                     logical(layerConnect), ...
                    %                     outputConnect);
                    % 
                    % deepnet.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
                    %                     'plotregression', 'plotfit'};
                    % for i=1:layer_n
                    %     if (i <= numel(nncfg.dnn.arch))
                    %         deepnet.layers{i}.dimensions = nncfg.dnn.arch(i);
                    %         deepnet.layers{i}.transferFcn = 'logsig';                        
                    %     else
                    %         deepnet.layers{i}.transferFcn = 'purelin';
                    %     end
                    %     deepnet.layers{i}.initFcn = 'initnw';
                    % end                
                    % 
                    % XX = self.nndbs{self.in_idx}.features(:, [self.tr_indices self.val_indices]);
                    % YY = self.nndbs{self.out_idx}.features(:, [self.tr_indices self.val_indices]);
                    % deepnet = configure(deepnet, XX, YY);
                    % 
                    % deepnet.adaptFcn = 'adaptwb';
                    % deepnet.inputWeights{1}.learnFcn = 'learngdm';
                    % deepnet.layerWeights{1}.learnFcn = 'learngdm';  

                    % Create a fitnet
                    deepnet = fitnet(nncfg.dnn.arch, nncfg.trainFcn);
                    deepnet.inputs{1}.processFcns = {};
                    deepnet.outputs{end}.processFcns = {};

                    deepnet.adaptFcn = '';
                    deepnet.inputWeights{1}.learnFcn = '';                
                    for i=1:numel(deepnet.layers)-1
                        deepnet.layerWeights{i+1, i}.learnFcn = '';                    
                        if (isfield(nncfg, 'transferFcn'))
                            deepnet.layers{i}.transferFcn = nncfg.transferFcn;
                        end
                    end


                    for i=1:numel(deepnet.layers)
                        if (isfield(nncfg, 'act_fns'))
                            deepnet.layers{i}.transferFcn = nncfg.act_fns{i};
                        end                        
                        if (isfield(nncfg, 'initFcn'))
                            deepnet.layers{i}.initFcn = nncfg.initFcn;
                        end
                    end                                    

                    % For a list of all processing functions type: help nnprocess
                    if (isfield(nncfg, 'inPreProcess'))
                        processFcns = cell(0, numel(nncfg.inPreProcess));
                        processParams = cell(0, numel(nncfg.inPreProcess));
                        for i=1:numel(nncfg.inPreProcess)
                            processFcns{i} = nncfg.inPreProcess{i}.fcn;
                            processParams{i} = nncfg.inPreProcess{i}.processParams;
                        end
                        deepnet.inputs{1}.processFcns = processFcns;
                        deepnet.inputs{1}.processParams = processParams;
                    end

                    if (isfield(nncfg, 'outPreProcess'))
                        processFcns = cell(0, numel(nncfg.outPreProcess));
                        processParams = cell(0, numel(nncfg.outPreProcess));
                        for i=1:numel(nncfg.outPreProcess)
                            processFcns{i} = nncfg.outPreProcess{i}.fcn;
                            processParams{i} = nncfg.outPreProcess{i}.processParams;
                        end
                        deepnet.outputs{end}.processFcns = processFcns;
                        deepnet.outputs{end}.processParams = processParams;
                    end

                % Basic Deep Autoencoder Network
                else
                    % arch = I [50 30 50] O => [x 1 2 3 4]
                    deepnet = fitnet(nncfg.arch, nncfg.trainFcn);                
                    deepnet.inputs{1}.processFcns = {};
                    deepnet.outputs{end}.processFcns = {};
                    deepnet = configure(deepnet, XX, YY);
                    deepnet.name = 'Basic DAE';               

                    % No of layers incluing input and output layers
                    n_layers = numel(nncfg.arch) + 2;                

                    for enc_i=1:aes_n
                        autoenc = pretr_nets{enc_i};
                        W_e = autoenc.EncoderWeights;
                        b_e = autoenc.EncoderBiases;

                        % Set encoder weights
                        if (enc_i == 1)
                            deepnet.IW{1} = W_e;
                            deepnet.b{1} = b_e;
                            deepnet.layers{enc_i}.name = 'ENC';
                            deepnet.layers{enc_i}.transferFcn = nncfg.act_fns{enc_i};
                        else
                            deepnet.LW{enc_i,enc_i-1} = W_e;
                            deepnet.b{enc_i} = b_e;
                            deepnet.layers{enc_i}.name = 'ENC';
                            deepnet.layers{enc_i}.transferFcn = nncfg.act_fns{enc_i};
                        end                    

                        % Set decoder weights
                        dec_i = n_layers - enc_i;
                        deepnet.LW{dec_i, dec_i-1} = W_e';
                        deepnet.b{dec_i} = zeros(size(deepnet.b{dec_i}));
                        deepnet.layers{dec_i}.name = 'DEC';
                        deepnet.layers{dec_i}.transferFcn = nncfg.act_fns{dec_i};
                    end
                    deepnet.layers{n_layers-1}.transferFcn = 'purelin';                
                end
            end
                        
            % Validate the deepnet to see the learnt weights and bias are transfered properly
            % self.validate_(deepnet, pretr_nets);                

            % Configure deepnet (rest of the configs are obtained from 'net' network as per matlab
            % documentation
            % Training function and related learning parameters
            % default: 'trainscg'
            deepnet.trainFcn = nncfg.trainFcn;
            fields = fieldnames(nncfg.trainParam);
            for fn=fields'
              % since fn is a 1-by-1 cell array, you still need to index into it, unfortunately
              deepnet.trainParam.(fn{1}) = nncfg.trainParam.(fn{1});
            end           

            % Set performance param
            deepnet.performFcn = nncfg.cost_fn;
            
            % Set reguralization ratio
            deepnet.performParam.regularization = nncfg.reg_ratio;
            
            % Setup Division of Data for Training, Validation, Testing
            if (self.split_val_data)
                deepnet.divideFcn = 'divideind';
                deepnet.divideMode = 'sample';  % Divide up every sample
                deepnet.divideParam.trainInd = self.tr_indices;
                deepnet.divideParam.valInd = self.val_indices;
                deepnet.divideParam.testInd = self.te_indices;                
            else                
                deepnet.divideFcn               = 'dividerand'; % Divide data randomly
                deepnet.divideMode              = 'sample';     % Divide up every sample
                deepnet.divideParam.trainRatio  = 70/100;
                deepnet.divideParam.valRatio    = 15/100;
                deepnet.divideParam.testRatio   = 15/100;                
            end

           	% Deep neural network
            if (isfield(nncfg, 'dnn'))
                deepnet = configure(deepnet, XX, YY);
            end
            
            % Fine-tune the deep network
            if (self.fine_tune)
                [deepnet_ft, tr_stat] = train(deepnet, XX, YY, 'useGPU', 'yes'); %, 'showResources', 'yes');
                total_time = total_time + sum(tr_stat.time);
                tr_stats{net_idx} = tr_stat;
            else 
                deepnet_ft = deepnet;
                
            end            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Test the Network
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Training dataset visualization
            PYY = deepnet_ft(XX);
            e = gsubtract(YY,PYY);
            tr_db_perf = perform(deepnet_ft, YY, PYY);           

            % Testing dataset visualization
            XX_te       = self.nndbs{self.in_idx}.features(:, self.te_indices);
            YY_te       = self.nndbs{self.out_idx}.features(:, self.te_indices);
            XX_val      = self.nndbs{self.in_idx}.features(:, self.val_indices);
            YY_val      = self.nndbs{self.out_idx}.features(:, self.val_indices);
            PYY_te      = deepnet_ft(XX_te);
            PYY_val     = deepnet_ft(XX_val);
            test_db_perf = perform(deepnet_ft, YY_te, PYY_te);
            val_db_perf = perform(deepnet_ft, YY_val, PYY_val);            
            e_te        = gsubtract(YY_te, PYY_te);
            e_val       = gsubtract(YY_val, PYY_val);
            
            r_tr  = regression(YY, PYY, 'one');
            r_te  = regression(YY_te, PYY_te, 'one');
            r_val = regression(YY_val, PYY_val, 'one');
            
            % Display performance
            disp(['DNN_ERR: Tr:' num2str(tr_db_perf) ' R:' num2str(r_tr) ...
                ' Te_EX:' num2str(test_db_perf) ' R:' num2str(r_te) ' S:(m:' num2str(mean(e_te(:))) ', d:' num2str(std(e_te(:))) ')' ...
                ' Val:' num2str(val_db_perf) ' R:' num2str(r_val) ' S:(m:' num2str(mean(e_val(:))) ', d:' num2str(std(e_val(:))) ')'...
                ' Time:' num2str(total_time/1000)]);
            
            if (~self.split_val_data)
                XX = self.nndbs{self.in_idx}.features;
                YY = self.nndbs{self.out_idx}.features;
            end
            
            % Callback
            self.on_train_end_(pp_infos, nncfg, pretr_nets, deepnet, deepnet_ft, tr_stats, XX, YY, PYY, XX_te, YY_te, PYY_te);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Abstract, Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected: Callbacks       
       	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        on_train_start_(self, pp_infos);        
        on_train_end_(self, pp_infos, nncfg, pretr_nets, deepnet, deepnet_ft, tr_stats, XX, YY, PYY, XX_te, YY_te, PYY_te);
        XX = on_dr_end_(self, pretr_nets, XX);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface       
       	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function [XXT, XXT_te, XXT_val] = on_rl_loop_init_(self, loop_idx)
            if (self.split_val_data)
                XXT = self.nndbs{self.out_idx}.features;
            else                
                XXT = self.nndbs{self.out_idx}.features(:, self.tr_indices);
            end                
            XXT_te = self.nndbs{self.out_idx}.features(:, self.te_indices);                
            XXT_val = self.nndbs{self.out_idx}.features(:, self.val_indices); 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function pp_infos = pp_map_min_max_(self, pp_infos)
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
                            
            % Perform mapminmax for the nndbs
            for i=1:numel(self.nndbs)
                pp_info = pp_infos{i};

                if (~isfield(pp_info, 'mapminmax'))
                    continue;
                end
                assert(~isempty(pp_info.mapminmax))
                min = pp_info.mapminmax(1);
                max = pp_info.mapminmax(2);

                nndb = self.nndbs{i};

                % mapminmax
                db = zeros(nndb.h*nndb.w*nndb.ch, nndb.n);

                tr_db = self.nndbs{i}.features(:, self.tr_indices);
                val_db = self.nndbs{i}.features(:, self.val_indices);
                te_db = self.nndbs{i}.features(:, self.te_indices);

                [tr_db, pp_infos{i}.int.setting] = mapminmax(tr_db, min, max);
                [val_db] = mapminmax('apply', val_db, pp_infos{i}.int.setting);
                [te_db] = mapminmax('apply', te_db, pp_infos{i}.int.setting);

                db(:, self.tr_indices) = tr_db;
                db(:, self.val_indices) = val_db;
                db(:, self.te_indices) = te_db;

                self.nndbs{i} = NNdb(['db' num2str(i) '_mapminmax'], db, nndb.n_per_class, false, nndb.cls_lbl, Format.H_N);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function validate_(self, deepnet, pretr_nets) 
            % VALIDATE: Assertion checks to see whether the pre-trained weights were properly
            % transfered.
            %
            
            if (isempty(pretr_nets))
                return
            end
                        
            % Bias check                
            for i=1:numel(pretr_nets)
                net = pretr_nets{i};
                
                if (isa(net,'Autoencoder'))
                    assert(isequal(deepnet.b{i}, pretr_nets{i}.network.b{1}));
                    
                    % Last network
                    if (i == numel(pretr_nets) && ~strcmp(deepnet.name, 'Basic DAE')) 
                        assert(isequal(deepnet.b{i+1}, pretr_nets{i}.network.b{2}));
                    end
                    
                else
                    assert(isequal(deepnet.b{i}, pretr_nets{i}.b{1}));
                            
                    % Last network
                    if (i == numel(pretr_nets) && ~strcmp(deepnet.name, 'Basic DAE')) 
                        assert(isequal(deepnet.b{i+1}, pretr_nets{i}.b{2}));
                    end
                    
                end
            end
            
            % Weights check
            assert(isequal(deepnet.IW{1}, pretr_nets{1}.network.IW{1}));                
            for i=2:numel(pretr_nets)
                net = pretr_nets{i};
                
                if (isa(net,'Autoencoder'))
                    assert(isequal(deepnet.LW{i, i-1}, pretr_nets{i}.network.IW{1}));
                    
                    % Last network
                    if (i == numel(pretr_nets) && ~strcmp(deepnet.name, 'Basic DAE')) 
                        assert(isequal(deepnet.LW{i+1, i}, pretr_nets{i}.network.LW{2, 1}));
                    end
                    
                else
                    assert(isequal(deepnet.LW{i, i-1}, pretr_nets{i}.IW{1}));
                    
                    % Last network
                    if (i == numel(pretr_nets) && ~strcmp(deepnet.name, 'Basic DAE')) 
                        assert(isequal(deepnet.LW{i+1, i}, pretr_nets{i}.LW{2, 1}));
                    end
                    
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [autoenc, tr_stat] = trainAutoencoder(self, X, Y, aecfg, varargin)
            % trainAutoencoder   Train an autoencoder
            %   autoenc = trainAutoencoder(x) returns a trained autoencoder where x is
            %   the training data.
            %
            %   x may be a matrix of training samples where each column represents a
            %   single sample, or alternatively it can be a cell array of images, where
            %   each image has the same number of dimensions.
            %
            %   autoenc = trainAutoencoder(x, hiddenSize) returns a trained autoencoder
            %   where x is the training data, and hiddenSize specifies the size of the 
            %   autoencoder's hidden representation. The default value of hiddenSize is
            %   10.
            %
            %   autoenc = trainAutoencoder(..., Name1, Value1, Name2, Value2, ...)
            %   returns a trained autoencoder with additional options specified by the
            %   following name/value pairs:
            %
            %       'EncoderTransferFunction' - The transfer function for the encoder.
            %                                   This can be either 'logsig' for the
            %                                   logistic sigmoid function, or 'satlin'
            %                                   for the positive saturating linear
            %                                   transfer function. The default is
            %                                   'logsig'. <= ONLY SUPPORT (or 'satlin')
            %       'DecoderTransferFunction' - The transfer function for the decoder.
            %                                   This can be 'logsig' for the logistic
            %                                   sigmoid function, 'satlin' for the
            %                                   positive saturating linear transfer
            %                                   function, or 'purelin' for the linear
            %                                   transfer function. The default is
            %                                   'logsig'. <= ONLY SUPPORT (or 'satlin', 'purelin')
            %       'MaxEpochs'               - The maximum number of training epochs. 
            %                                   The default is 1000.
            %       'L2WeightRegularization'  - The coefficient that controls the
            %                                   weighting of the L2 weight regularizer.
            %                                   The default value is 0.001.
            %       'LossFunction'            - The loss function that is used for
            %                                   training. The default is 'msesparse'. <=ONLY SUPPORT
            %       'ShowProgressWindow'      - Indicates whether the training window 
            %                                   should be shown during training. The 
            %                                   default is true.
            %       'SparsityProportion'      - The desired proportion of training
            %                                   examples which a neuron in the hidden 
            %                                   layer of the autoencoder should 
            %                                   activate in response to. Must be 
            %                                   between 0 and 1. A low value encourages
            %                                   a higher degree of sparsity. The 
            %                                   default is 0.05.
            %       'SparsityRegularization'  - The coefficient that controls the
            %                                   weighting of the sparsity regularizer.
            %                                   The default is 1.
            %       'TrainingAlgorithm'       - The training algorithm used to train
            %                                   the autoencoder. Only the value
            %                                   'trainscg' for scaled conjugate
            %                                   gradient descent is allowed, which is
            %                                   the default.
            %       'ScaleData'               - True when the autoencoder rescales the 
            %                                   input data. The default is true.
            %       'UseGPU'                  - True if the GPU is used for training.
            %                                   The default value is false.
            %
            %   Example 1: 
            %       Train a sparse autoencoder to compress and reconstruct abalone
            %       shell ring data, and then measure the mean squared reconstruction
            %       error.
            %
            %       x = abalone_dataset;
            %       hiddenSize = 4;
            %       autoenc = trainAutoencoder(x, hiddenSize, ...
            %           'MaxEpochs', 400, ...
            %           'DecoderTransferFunction', 'purelin');
            %       xReconstructed = predict(autoenc, x);
            %       mseError = mse(x-xReconstructed)
            %
            %   Example 2:
            %       Train a sparse autoencoder on images of handwritten digits to learn
            %       features, and use it to compress and reconstruct these images. View
            %       some of the original images along with their reconstructed
            %       versions.
            %
            %       x = digitsmall_dataset;
            %       hiddenSize = 40;
            %       autoenc = trainAutoencoder(x, hiddenSize, ...
            %           'L2WeightRegularization', 0.004, ...
            %           'SparsityRegularization', 4, ...
            %           'SparsityProportion', 0.15);
            %       xReconstructed = predict(autoenc, x);
            %       figure;
            %       for i = 1:20
            %           subplot(4,5,i);
            %           imshow(x{i});
            %       end
            %       figure;
            %       for i = 1:20
            %           subplot(4,5,i);
            %           imshow(xReconstructed{i});
            %       end
            %
            %   See also Autoencoder

            %   Copyright 2015 The MathWorks, Inc.

            paramsStruct  = Autoencoder.parseInputArguments(varargin{:});
            autonet = Autoencoder.createNetwork(paramsStruct);
            autonet.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotregression', 'plotfit'};

            % Training function and related learning parameters
            % default: 'trainscg'
            autonet.trainFcn = aecfg.trainFcn;
            fields = fieldnames(aecfg.trainParam);
            for fn=fields'
              %# since fn is a 1-by-1 cell array, you still need to index into it, unfortunately
              autonet.trainParam.(fn{1}) = aecfg.trainParam.(fn{1});
            end

            % traingdm learning parameters
            % net.trainParam.epochs = 5000;
            % net.trainParam.mc = 0.9;
            % net.trainParam.lr = 1;
            
            % For a list of all training functions type: help nntrain
            % 'trainlm' is usually fastest.
            % 'trainbr' takes longer but may be better for challenging problems.
            % 'trainscg' uses less memory. Suitable in low memory situations.
            % 'trainlm';  % Levenberg-Marquardt backpropagation.
            
            % Setup Division of Data for Training, Validation, Testing
            if (self.split_val_data)
                autonet.divideFcn = 'divideind';
                autonet.divideMode = 'sample';  % Divide up every sample
                autonet.divideParam.trainInd = self.tr_indices;
                autonet.divideParam.valInd = self.val_indices;
                autonet.divideParam.testInd = self.te_indices;              
            else                
                autonet.divideFcn = 'dividerand';  % Divide data randomly
                autonet.divideMode = 'sample';  % Divide up every sample
                autonet.divideParam.trainRatio  = 85/100;
                autonet.divideParam.valRatio    = 15/100;
                autonet.divideParam.testRatio   = 0;
            end
            
            %if (~strcmp(aecfg.cost_fn, 'msesparse'))
                % trainAutoencoder only supports 'msesparse' by default. Thus customized.
                autonet.performFcn = aecfg.cost_fn; % will also reset performParams
                autonet.layers{1}.transferFcn = aecfg.enc_fn;
                autonet.layers{2}.transferFcn = aecfg.dec_fn;
                 
                if (aecfg.is_sparse) % [FUTURE_PROOF] For sparse cost_fns other than msesparse                
                    % Set sparse related parameters
                    autonet.performParam.L2WeightRegularization = aecfg.l2_wd;
                    autonet.performParam.sparsityRegularization = aecfg.sparse_reg;
                    autonet.performParam.sparsity = aecfg.sparsity;
                end
            %end
            
            % Set regularization ratio (cannot set a value for sparse auto encoders): TODO: BUG
            autonet.performParam.regularization = aecfg.reg_ratio;
            
            % For a list of all processing functions type: help nnprocess
            processFcns = cell(0, numel(aecfg.inPreProcess));
            processParams = cell(0, numel(aecfg.inPreProcess));
            for i=1:numel(aecfg.inPreProcess)
                processFcns{i} = aecfg.inPreProcess{i}.fcn;
                processParams{i} = aecfg.inPreProcess{i}.processParams;
            end
            autonet.inputs{1}.processFcns = processFcns;
            autonet.inputs{1}.processParams = processParams;

            processFcns = cell(0, numel(aecfg.outPreProcess));
            processParams = cell(0, numel(aecfg.outPreProcess));
            for i=1:numel(aecfg.outPreProcess)
                processFcns{i} = aecfg.outPreProcess{i}.fcn;
                processParams{i} = aecfg.outPreProcess{i}.processParams;
            end
            autonet.outputs{2}.processFcns = processFcns;
            autonet.outputs{2}.processParams = processParams;
            
            % Layer weight init function
            autonet.layers{1}.initFcn = aecfg.encoderInitFcn;
            autonet.layers{2}.initFcn = aecfg.decoderInitFcn;
            
            % Initialize the weights to be compatible within the range of 'satlin' function
            if (strcmp(autonet.layers{1}.transferFcn, 'satlin'))
                % https://au.mathworks.com/help/nnet/ref/initnw.html
                % https://au.mathworks.com/help/nnet/ref/init.html (to check the init W, b values)
                % autonet = configure(autonet, X, X);
                % autonet.b or autonet.IW or autonet.LW
                % r = autonet.IW{1} * X + repmat(autonet.b{1}, 1, 3188);
                % s = satlin(r);
                % autonet = init(autonet); to reinitialize

                % Encoder weights
                %autonet.layers{1}.initFcn = 'initnw';
                %autonet.inputWeights{1}.initFcn = '';

                % Clear the weights and initilize for satlin activation.
                autonet.layers{1}.initFcn = '';
                autonet.inputWeights{1}.initFcn = '';
                
                if (~isempty(Y))
                    autonet = configure(autonet, X, Y);
                else
                    autonet = configure(autonet, X, X);
                end
                
                I = size(X, 1);
                H = autonet.layers{1}.size;
                % O = autonet.layers{1}.size;
                
                % Peform explicit initialization
                IW = 0.21*(0.1 + randsmall(H,I));
                b1 = 0.21*(0.1 + randsmall(H,1));
                % LW = 0.1 + randsmall(O,H);
                % b2 = 0.1 + randsmall(O,1);
                autonet.IW{1,1} = IW;
                autonet.b{1,1} = b1;
                % autonet.LW{2,1} = LW;
                % autonet.b{2,1} = b2;
            end
             
            % Train AE
            if (~isempty(Y))
                %autonet = self.batch_train(autonet, X, Y, 125000, 5000)
                [autonet, tr_stat] = train(autonet,X, Y, 'useGPU','yes');
                autoenc = Autoencoder(autonet, true, 1,0);
                
            else
                assert(false); % This path is deprecated with the introduction of tied weights
                % Train Autoencoder to reduce the dimensionality
                tr_stat = [];
                autoenc = Autoencoder.train(X, autonet, paramsStruct.UseGPU);
            end 
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function net = batch_train_(self, net, X, Y, epochs, batchSize)
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;
            net.trainParam.show = 1;

            N = size(X, 2);
            batchStart = 1:batchSize:N;
            idx = randperm(N);
            break_out = false;
            
            for epoch = 1:epochs
                if (mod(epoch, 200) == 0)
                    net.trainParam.showCommandLine = true;
                    disp(['Epoch:' num2str(epoch)])
                else
                    net.trainParam.showCommandLine = false;
                end
                
                for k = 1:length(batchStart)                    
                    bs = batchStart(k);                    
                    cur_batch_size = batchSize;
                    
                    if (bs+batchSize-1 > N)
                        cur_batch_size = N - bs + 1;
                    end

                    bidx = idx(bs:bs+cur_batch_size-1);
                    net.trainParam.epochs = 1;

                    % Train Autoencoder with a target
                    [net, tr] = train(net,X(:, bidx),Y(:, bidx), 'useGPU','yes');
                    net.trainParam.showCommandLine = false;
                    if (~strcmp(tr.stop, 'Maximum epoch reached.'))
                        break_out = true;
                        break;
                    end

            %         [ autotest , tr ] = train( autotest , ...
            %             X(:,bidx) , X(:,bidx) ) ;
            %         pr(epoch,k) = tr.perf(end);
            
                    % Epoch:6400
                    % Calculation mode: GPU
                    % 
                    % Training Autoencoder with TRAINSCG.
                    % Epoch 0/1, Time 0.001, Performance 0.00073646/1e-10, Gradient 0.00062527/1e-10, Validation Checks 0/2100
                    % Epoch 1/1, Time 0.022, Performance 0.00073624/1e-10, Gradient 0.00021066/1e-10, Validation Checks 0/2100
                    % Training with TRAINSCG completed: Maximum epoch reached.
                end
                
                if (break_out)
                    break
                end

                %fprintf( 'epoch:%d , performence : %.2f\n' , epoch , pr(epoch,end) ) ;
            end                             
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end