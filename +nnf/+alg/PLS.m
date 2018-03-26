classdef PLS
    %PLS: Partial Least Squares algorithm and variants.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - PLS.l2     
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties        
    end
    
    methods (Access = public, Static) 
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [W_X, W_Y] = l2(nndb_tr, nndb_tr_out, info)
            % L2: leanrs the PLS subspace with l2-norm.
            %   IMOPORTANT:
            %       When (info.alg == 'PLS_BASES')
            %       -----------------------------
            %       COFx = W'*X;    => W_X' * X
            %       COFy = Z\Y;     => W_Y \ Y  
            %       Refer PLS_Bases.m (Author's Code)
            %
            % Parameters
            % ----------
            % nndb_tr : nnf.db.NNdb
            %     Data object that contains the training database.
            % 
            % nndb_tr : nnf.db.NNdb
            %     Data object that contains the training target database.
            %
            % info : struct, optional
            %     Provide additional information to perform PCA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.bases = 0;     % No of bases required (0 = keep all)  
            % 
            % 
            % Returns
            % -------
            % W_tr : 2D matrix -double
            %     Projecttion matrix for training database.
         	%
            % W_tr_out : 2D matrix -double
            %     Projecttion matrix for training target database.
            %
            %
            % Examples
            % --------
            % import nnf.alg.PLS;
            % info.bases = 10;
            % [W_tr, W_tr_out] = PLS.l2(nndb_tr, nndb_tr_out, info);
            %
            
            % Imports 
            import nnf.alg.PLS;
            
            % Set defaults for arguments
            if (nargin < 3), info = []; end
            
            % Set defaults for info fields, if the field does not exist
            if (~isfield(info,'alg')); info.alg = 'PLS_BASES'; end
            if (~isfield(info,'bases')); info.bases = 0; end
            
            if (info.bases == 0)
                % Since PLS always finds a space whose rank is lower to the lowest rank of provided
                % databases
                m_tr = min(nndb_tr.h*nndb_tr.w, nndb_tr.n);
                m_tr_out = min(nndb_tr_out.h*nndb_tr_out.w, nndb_tr_out.n);
                info.bases = min(m_tr, m_tr_out);
            end
            
            if (strcmp(info.alg, 'PLS_BASES'))
                % Optimization 1
                [W_X, W_Y] = PLS_Bases(nndb_tr.features, nndb_tr_out.features, info.bases);
                
                % IMOPORTANT: Refer PLS_Bases.m (Author's Code)
                % COFx = W'*X;    => W_X' * X
                % COFy = Z\Y;     => W_Y \ Y           
                
            elseif (strcmp(info.alg, 'PLS_SVD'))
                % Optimization 2
                [W_X, W_Y] = PLS.optimize(nndb_tr.features, nndb_tr_out.features, info.bases);
                
            elseif (strcmp(info.alg, 'PLS_NIPALS'))
                % Optimization 3
                [W_X, W_Y] = PLS.plsnipals(nndb_tr.features', nndb_tr_out.features', info.bases);
                
            elseif (strcmp(info.alg, 'PLS_MATLAB'))
                % Optimization 4 (TODO: CHECK, RECOGNITION FAILS) - MATLAB plsregress
                [W_X, W_Y] = plsregress(nndb_tr.features', nndb_tr_out.features', info.bases);
                
            end
        end
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public, Static)
      	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [P, Q] = optimize(X, Y, bases)
            % %
            % % NIPALS: Iterative algorithm
            % for n = 1:bases;
            % 
            %     u = Y(:,1);
            %     t1=X(:,1);
            %     t2=X(:,2);
            % 
            % 
            %     while norm(t1-t2) > 1e-6
            %     t1 = t2;
            %     p = X'*u/norm(X'*u);
            %     t2 = X*p;
            %     q = Y'*t2/norm(Y'*t2);
            %     u = Y*q;
            %     end
            % 
            %     T(n,:) = t2;
            %     U(n,:) = u;
            %     P(n,:) = p;
            %     Q(n,:) = q;
            % 
            %     X = X-t2*p';
            %     Y = Y-u*q';
            % 
            % end
            
            % Decomposition of cross co-variance matrices.           
            [~, ~, Q] = svd((X*Y')'*(X*Y'));
            [~, ~, P] = svd((Y*X')'*(Y*X'));
            Q = Q(:,1:bases);
            P = P(:,1:bases);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [P, Q] = plsnipals(X,Y, bases)
            % Berny's implementation
            %+++ The NIPALS algorithm
            % REF: https://www.ibm.com/support/knowledgecenter/en/SSLVMB_21.0.0/com.ibm.spss.statistics.help/alg_pls_estimation_nipals.htm
            for i=1:bases
                error = 1;
                u = Y(:,1);
                niter = 0;
                while (error>1e-8 && niter<10000)  % for convergence test
                    p = X'*u/ norm(X'*u);
                    t = X*p;
                    q = Y'*t/norm(Y'*t);
                    u1= Y*q;
                    error = norm(u1-u)/norm(u);
                    u = u1;
                    niter = niter + 1;
                end
                X = X - t*p';
                Y = Y - u1*q'; % t * p'
                
                %+++ store
                P(:,i) = p;
                Q(:,i) = q;                
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
                
end

