classdef KDA < nnf.alg.LDA
    %KDA Summary of this class goes here
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - KDA.l2 (regularized or SVD L2)
    
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
            %     Provide additional information to perform KDA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.Regu      = false;    % Solve the sinularity problem by SVD
            %     inf.ReguAlpha = 0.1;      % Regularizaion paramaeter (if Regu = true)
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
            % import nnf.alg.KDA;
            % [W, ki] = KDA.l2(nndb_tr)
            % 
                        
            % Imports 
            import nnf.alg.KDA;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end
                                
            % Fetch eigen faces
            [W, kinfo] = KDA.kfisher_face_core(nndb.features, nndb.cls_lbl, info);            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    end
    
    methods (Access = private, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [kffaces, kinfo] = kfisher_face_core(X, XL, info) 
            %KEIG_FACE_CORE performs kernel eigen face learning.
            
            % Set defaults for info fields, if the field does not exist   
            if (~isfield(info,'Regu')); info.Regu = false; end;
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
                        
            % Set defaults for info fields, depending on Regu property value
            if (info.Regu)
                
                % Set defaults for info fields, if the field does not exist
                if (~isfield(info,'ReguAlpha')); info.ReguAlpha = 0.1; end;  
                
                options.Regu = info.Regu;
                options.ReguAlpha = info.ReguAlpha;
            end
            
            % Perform KDA
            [kffaces, ~] = KDA(options, XL, X');      
            
            % Set kernel info struct for later use
            kinfo.koptions = options;
            kinfo.use_kda = true;            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

