classdef SRC
    %SRC: Sparse Coding algorithm and varients.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - SRC.l1
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [accuracy] = l1(nndb_g, nndb_p, info) 
            % L1: performs SRC with l1-norm     
            % TODO: return sparse coefficient and perform the recognition in Util class
            %
            % Parameters
            % ----------
            % nndb_g : nnf.db.NNdb
            %     Data object that to be used as the dictionary.
            %
            % nndb_p : nnf.db.NNdb
            %     Data object that to be used as the targets.
            %
            % info : struct, optional
            %     Provide additional information to perform KDA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.lambda = 0.01;  % the parameter to optimization algorithm l1_ls
            %
            %
            % Returns
            % -------
            % accuracy : double
            %     Accuracy of classification.
         	%
            %
            % Examples
            % --------
            % import nnf.alg.SRC;
            % accuracy = SRC.l1(nndb_g, nndb_p)
            % 
            
            % Set defaults for arguments
            if (nargin < 3), info = []; end
            
            % Set defaults for info fields, if the field does not exist  
            if (~isfield(info,'lambda')); info.lambda = 0.01; end; 
            
            %
            % Model selection (with different lambda)
            % [param,maxAcc]=lineSearchSRC2(trainSet,trainClass);
            % option.lambda=param;
            %
            
            % TODO: Troubleshoot POSTIVE DEFNITE error 
            % optionSRC1.rubost = true; % SRC1
            %[testClassPredicted, sparsity, otherOutput] = src(trainSet, nndb_g.cls_lbl', testSet, optionSRC2);
            
            % SRC2 for overcomplete representations
            optionSRC2.lambda = info.lambda;
            optionSRC2.method = 'interiorPoint';

            % Normalization, before applying SRC algorihm
            [testClassPredicted, sparsity, otherOutput] = ...
                SRC2(normc(nndb_g.features), nndb_g.cls_lbl', normc(nndb_p.features), optionSRC2);
            
            % Addition information on residuals
            % residualSRC2 = otherOutput; % regression residuals
            
            % Calculate accuracy
            accuracy = (sum(testClassPredicted == nndb_p.cls_lbl')/nndb_p.n)* 100;                        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function SRC_MCC()
            % TODO: SLINDA
        end
    end
    
    methods (Access = private, Static) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % IMPLEMENTATION 1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function [A, B] = PreProcess(A, B, pca)
        % TODO: Remove this method and incorporate it to NNdb class
            [~,d1] = size(A);
            [~,d2] = size(B);

            m = mean(A, 2);
            A = A - repmat(m, [1,d1]);
            B = B - repmat(m, [1,d2]);

            A = src_normalize(A);
            B = src_normalize(B);

            if (pca > 0)
                v = zPCA(A, pca); %? missing
                A = v'*A;
                B = v'*B;
            end

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function x = src_lasso(D, B, lamda, pca, error) 
       
            if (error) % For handling the occlusion or error with SRC
                dim1 = size(D, 1);
                D = [D eye(dim1)]; 
            end 

            [D, B] = PreProcess(D, B, pca);
            
            te_n = size(B, 2);
            x = zeros(size(D,2), te_n);
            for i = 1 : te_n
                x(:, i) = lasso_ADMM(D, B(:,i), lamda, 1, 1);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [z, history] = lasso_ADMM(A, b, lambda, rho, alpha) 
            % lasso  Solve lasso problem via ADMM
            %
            % [z, history] = lasso(A, b, lambda, rho, alpha);
            % 
            % Solves the following problem via ADMM:
            %
            %   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
            %
            % The solution is returned in the vector x.
            %
            % history is a structure that contains the objective value, the primal and 
            % dual residual norms, and the tolerances for the primal and dual residual 
            % norms at each iteration.
            % 
            % rho is the augmented Lagrangian parameter. 
            %
            % alpha is the over-relaxation parameter (typical values for alpha are 
            % between 1.0 and 1.8).
            %
            %
            % More information can be found in the paper linked at:
            % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
            %

            t_start = tic;

            %% Global constants and defaults

            QUIET    = 1;
            MAX_ITER = 1000;
            ABSTOL   = 1e-4;
            RELTOL   = 1e-2;

            %% Data preprocessing

            [m, n] = size(A);

            % save a matrix-vector multiply
            Atb = A'*b;

            %% ADMM solver

            x = zeros(n,1);
            z = zeros(n,1);
            u = zeros(n,1);

            % cache the factorization
            [L U] = factor(A, rho);

            if ~QUIET
                fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
                  'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
            end

            for k = 1:MAX_ITER

                % x-update
                q = Atb + rho*(z - u);    % temporary value
                if( m >= n )    % if skinny
                   x = U \ (L \ q);
                else            % if fat
                   x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
                end

                % z-update with relaxation
                zold = z;
                x_hat = alpha*x + (1 - alpha)*zold;
                z = shrinkage(x_hat + u, lambda/rho);

                % u-update
                u = u + (x_hat - z);

                % diagnostics, reporting, termination checks
                history.objval(k)  = objective(A, b, lambda, x, z);

                history.r_norm(k)  = norm(x - z);
                history.s_norm(k)  = norm(-rho*(z - zold));

                history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
                history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

                if ~QUIET
                    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
                        history.r_norm(k), history.eps_pri(k), ...
                        history.s_norm(k), history.eps_dual(k), history.objval(k));
                end

                if (history.r_norm(k) < history.eps_pri(k) && ...
                   history.s_norm(k) < history.eps_dual(k))
                     break;
                end

            end

            if ~QUIET
                toc(t_start);
            end

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function p = objective(A, b, lambda, x, z) 
            p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function z = shrinkage(x, kappa) 
            z = max( 0, x - kappa ) - max( 0, -x - kappa );
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [L U] = factor(A, rho) 
            [m, n] = size(A);
            if ( m >= n )    % if skinny
               L = chol( A'*A + rho*speye(n), 'lower' );
            else            % if fat
               L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
            end

            % force matlab to recognize the upper / lower triangular structure
            L = sparse(L);
            U = sparse(L');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % IMPLEMENTATION 1 END
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

