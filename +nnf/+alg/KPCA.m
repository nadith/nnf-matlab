classdef KPCA < nnf.alg.PCA
    %KPCA Summary of this class goes here
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - KPCA.l2
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
      	function [W, kinfo] = l2(nndb, info) 
            % L2: leanrs the KPCA subspace with l2-norm     
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
            %     inf.ReducedDim           = 0;            % No of dimension (0 = keep all)
            %     inf.KernelType           = 'Gaussian';   % e^{-(|x-y|^2)/2t^2}
            %                              = 'Polynomial'; % (x'*y)^d   
            %                              = 'PolyPlus'    % (x'*y+1)^d
            %                              = 'Linear'      % x'*y
            %     inf.t                    = 500;          % Kernel size (if inf.KernelType = 'Gaussian')
            %     inf.d                    = 2;            % Ploy factor (if inf.KernelType = 'Polynomial')
            %
            %
            % Returns
            % -------
            % W : 2D matrix -double
            %     Projecttion matrix.
         	%
            % kinfo : struct
            %     Kernel info structure that need to be provided along with W.
         	%
            %
            % Examples
            % --------
            % import nnf.alg.KPCA;
            % [W, ki] = KPCA.l2(nndb_tr, info)
            % 
                        
            % Imports 
            import nnf.alg.KPCA;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end
                                
            % Fetch eigen faces
            [W, kinfo] = KPCA.keig_face_core(nndb.features, info);            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [keig_faces, kinfo] = keig_face_core(X, info) 
            %KEIG_FACE_CORE performs kernel eigen face learning.
            
            % Set defaults for info fields, if the field does not exist   
            if (~isfield(info,'ReducedDim')); info.ReducedDim = 0; end;  % Keep all
            if (~isfield(info,'KernelType')); info.KernelType = 'Gaussian'; end;
      
            % Set defaults for info fields, depending on kernel type
            if (strcmp(info.KernelType, 'Gaussian'))                
                
                % Set defaults for info fields, if the field does not exist
                if (~isfield(info,'t')); info.t = 500; end;
                
                options.KernelType = info.KernelType;
                options.t = info.t;
                
            elseif (strcmp(info.KernelType, 'Polynomial'))
                
                % Set defaults for info fields, if the field does not exist
                if (~isfield(info,'d')); info.d = 2; end;
                
                options.KernelType = info.KernelType;
                options.d = info.d;                
            end
            
            % Perform KPCA
            options.ReducedDim = info.ReducedDim;
            [keig_faces, ~] = KPCA(X', options);
            
            % Set kernel info struct for later use
            kinfo.koptions = options;
            kinfo.use_kpca = true;
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

