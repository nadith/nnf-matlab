classdef TSNE
    %TSNE: T-Distributed Sochastic Neigbourhood Embedding algorithm and varients.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - TSNE.do
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndb_tsne = do(nndb, info)
            % DO: performs TSNE.
            %
            % Parameters
            % ----------
            % nndb : nnf.db.NNdb
            %     Data object that contains the database.
            % 
            % info : struct, optional
            %     Provide additional information to perform PCA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     info.ReducedDim    = 2;    % No of dimension needed via TSNE
            %     info.initial_dims  = 30;   % Initial dimension reduction via PCA
            %     info.perplexity    = 30;   % Perplexity of the Gaussian kernel
            %     info.max_iter      = 1000; % Maximum iteration steps for gradient descent opt.
            %
            %
            % Returns
            % -------
            % nndb_tsne : nnf.db.NNdb
            %     Reduced dimensional database.
            %
            %
            % Examples
            % --------
            % import nnf.alg.TSNE;
            % info.ReducedDim = 2;
            % info.initial_dims = 100;
            % info.perplexity = 5;
            % nndb_tsne = TSNE.do(nndb_tr, info);
            % figure, nndb_tsne.plot();
            %            
            
            % Imports
            import nnf.libs.DRToolBox;
            import nnf.db.NNdb;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end            
            if (~isfield(info,'ReducedDim')); info.ReducedDim = 2; end   
            if (~isfield(info,'initial_dims')); info.initial_dims = 30; end
            if (~isfield(info,'perplexity')); info.perplexity = 30; end
            if (~isfield(info,'max_iter')); info.max_iter = 1000; end
            
            % Perform TSNE with available infomation            
            mappedX = DRToolBox.tsne(nndb.features', nndb.cls_lbl, ...
                    info.ReducedDim, info.initial_dims, info.perplexity, info.max_iter);         
            
            % Transpose to column major matrix
            mappedX = mappedX';
            
            % Return
            fsize = size(mappedX, 1);
            nndb_tsne = ...
                NNdb('TSNE', reshape(mappedX, fsize, 1, 1, nndb.n), nndb.n_per_class, false, nndb.cls_lbl);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

