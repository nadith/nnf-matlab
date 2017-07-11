classdef AECfg
    %AECFG: Autoencoder configuration for pre-training layers
    %   Detailed explanation goes here
    
    properties (SetAccess = public) 
        hidden_nodes;
        enc_fn;        
        dec_fn;
        cost_fn;        
        
        % msesparse cost function params        
        sparsity;   % SparsityProportion
        sparse_reg; % SparsityRegularization
        l2_wd;      % L2WeightRegularization
        
        % mse, msesparse cost function params
        reg_ratio;  % 'regularization' can be set to any value between 0 and 1.
                    % The greater the regularization value, the more squared weights and biases 
                    % are included in the performance calculation relative to errors. The 
                    % default is 0, corresponding to no regularization. (feedforward = 0.01)
            
        inPreProcess;   % Input processing function and related parameters
                        %   pp.fcn = 'mapminmax';
                        %   pp.processParams.ymin = 0;
                        %   pp.processParams.ymax = 1;
                        %   self.inPreProcess = {pp};
        outPreProcess;  % Output processing function and related parameters
                        %   pp.fcn = 'mapminmax';
                        %   pp.processParams.ymin = 0;
                        %   pp.processParams.ymax = 1;
                        %   self.outPreProcess = {pp};
                        
        encoderInitFcn; % Encoder weight initilization function
        decoderInitFcn; % Decoder weight initilization function
        trainFcn;       % Train function
        trainParam;     % Train function related parameters
    end
    
    properties (SetAccess = public, Dependent)  
        is_sparse;  % Decide on the choice of 'cost_fn'
    end

    methods (Access = public) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = AECfg(nodes)           
            self.hidden_nodes = nodes;
            self.enc_fn = 'satlin';
            self.dec_fn = 'purelin';
            self.cost_fn = 'msesparse';
            self.sparsity = 0.01;
            self.sparse_reg = 1;
            self.l2_wd = 0.001;
            self.reg_ratio = 0;
            self.inPreProcess = cell(0, 0);
            self.outPreProcess = cell(0, 0);
            self.encoderInitFcn = 'initwb';
            self.decoderInitFcn = 'initwb';
            self.trainFcn = 'trainscg';
            self.trainParam.goal = 1e-10;
            self.trainParam.sigma = 1e-4; %1e-6;
            self.trainParam.lambda = 1e-6; %1e-8;
            self.trainParam.epochs = 525000;
            self.trainParam.max_fail = 2500;
            self.trainParam.min_grad = 1e-10; %5e-7;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = validate(self)
            success = true;
            
            if (self.is_sparse)
                if (~(strcmp(self.enc_fn, 'satlin') || ...
                        strcmp(self.enc_fn, 'logsig')))
                    error('SpraseAE: Only compatible with satlin, logsig transfer fn [0-1 range].');
                    success = false;
                end
            end            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.is_sparse(self)
            value = strcmp(self.cost_fn, 'msesparse');
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end 
end