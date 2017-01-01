classdef LLE    
    %LLE: Linear Local Embedding algorithm and varients.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - LLE.do
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndb_lle, mapping] = do(nndb, info)
            % LLE: performs linear local embedding.
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
            %     info.ReducedDim   = 2;        % No of dimension needed via LLE
            %     info.k            = 12;       % K neibours
            %     info.eig_impl     = 'Matlab'; % Eigen value decomposition implementaiton
            %
            %
            % Returns
            % -------
            % nndb_lle : nnf.db.NNdb
            %     Reduced dimensional database.
            %
            %
            % Examples
            % --------
            % import nnf.alg.LLE;
            % info.ReducedDim = 2;
            % info.k = 100;
            % nndb_lle = LLE.do(nndb_tr, info);
            % figure, nndb_lle.plot();
            %
            
            % Imports
            import nnf.libs.DRToolBox;
            import nnf.db.NNdb;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end
            
            % Perform LLE with available infomation   
            if (~isfield(info,'ReducedDim'))
                [mappedX, mapping] = DRToolBox.lle(nndb.features');
                
            elseif (~isfield(info,'k'))
                [mappedX, mapping] = DRToolBox.lle(nndb.features', info.ReducedDim);
            
            elseif (~isfield(info,'eig_impl'))
                [mappedX, mapping] = DRToolBox.lle(nndb.features', info.ReducedDim, info.k);
                
            else
                [mappedX, mapping] = DRToolBox.lle(nndb.features', info.ReducedDim, info.k, info.eig_impl);    
                
            end  
            
            % Transpose to column major matrix
            mappedX = mappedX';
            
            % Return
            fsize = size(mappedX, 1);
            nndb_lle = ...
                NNdb('LLE', reshape(mappedX, fsize, 1, 1, nndb.n), nndb.n_per_class, false, nndb.cls_lbl);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

