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
        function [accuracy, dist] = l1(nndb_g, nndb_p, info) 
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
            
            % Imports
            import nnf.alg.SRC;
            
            % Set defaults for arguments
            if (nargin < 3), info = []; end
            
            % Set defaults for info fields, if the field does not exist  
            if (~isfield(info,'lambda')); info.lambda = 0.01; end; 
            
            coeff = SRC.train_lasso(nndb_g.features, nndb_p.features, info.lambda, 0, 0);
            [accuracy, ~, dist] = SRC.lass_recogn(nndb_g.features, nndb_p.features, nndb_g.cls_lbl, nndb_p.cls_lbl, coeff, 2, 0, 0);
            dist = dist';
               
             
            % eye_ex = eye(nndb_g.cls_n);
            % eye_ex = repmat(eye_ex', 1, double(unique(nndb_g.n_per_class)))';
            % eye_ex = reshape(eye_ex, nndb_g.cls_n, []);
            % 
            % dist = dist * eye_ex;
            % 
            % [~, indices] = min(dist, [], 2);
            % assert(numel(indices) == nndb_p.n);
            % 
            % % To store the verification result
            % v = uint8(zeros(1, nndb_p.n));
            % 
            % for te_idx=1:nndb_p.n
            %     idx = indices(te_idx);
            % 
            %     % Set the verification result
            %     v(te_idx) = ((nndb_g.cls_lbl(idx) == nndb_p.cls_lbl(te_idx)));
            % end
            % 
            % % Calculate accuracy
            % accuracy = (sum(v) / nndb_p.n) * 100;
            
            
            
            
            
            
            
            % TODO:
            %
            % Model selection (with different lambda)
            % [param,maxAcc]=lineSearchSRC2(trainSet,trainClass);
            % option.lambda=param;
            %
            
%             % TODO: Troubleshoot POSTIVE DEFNITE error 
%             % optionSRC1.rubost = true; % SRC1
%             %[testClassPredicted, sparsity, otherOutput] = src(trainSet, nndb_g.cls_lbl', testSet, optionSRC2);
%             
%             % SRC2 for overcomplete representations
%             optionSRC2.lambda = info.lambda;
%             optionSRC2.method = 'interiorPoint';
% 
%             % Normalization, before applying SRC algorihm
%             [testClassPredicted, sparsity, otherOutput] = ...
%                 SRC2(normc(nndb_g.features), nndb_g.cls_lbl', normc(nndb_p.features), optionSRC2);
%             
%             % Addition information on residuals
%             % residualSRC2 = otherOutput; % regression residuals
%             
%             % Calculate accuracy
%             accuracy = (sum(testClassPredicted == nndb_p.cls_lbl')/nndb_p.n)* 100;                        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function SRC_MCC()
            % TODO: SLINDA
        end
    end
    
    methods (Access = private, Static) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % IMPLEMENTATION 1:
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function X = src_normalize(X)
            temp = sqrt(sum(X.^2,1));
            for i = 1 : size(X,2)
                X(:,i) = X(:,i)/temp(i);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ lamda ] = find_lambda(R, E, gr)
            [nTestSmp] = size(E, 2);
            groups = unique(gr);
            numGroups = length(groups);

            lamda = zeros(nTestSmp,1);

            for i = 1 : nTestSmp
                y = E(:,i);
                R_group_norm = zeros(numGroups, 1);
                for j = 1 : numGroups
                    temp = R(:,gr == groups(j));

                    %R_group_norm(j) = norm(temp'*y); %6 / 7 / 2012
                    R_group_norm(j) = max(abs(temp'*y));
                end
                lamda(i) = max(R_group_norm);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [A, B] = pre_process(A, B, pca)
            % TODO: Remove this method and incorporate it to NNdb class
            [~,d1] = size(A);
            [~,d2] = size(B);

            m = mean(A, 2);
            A = A - repmat(m, [1,d1]);
            B = B - repmat(m, [1,d2]);

            % Imports
            import nnf.alg.SRC;
            
            A = SRC.src_normalize(A);
            B = SRC.src_normalize(B);

            if (pca > 0)
                v = zPCA(A, pca); %? missing
                A = v'*A;
                B = v'*B;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function x = train_lasso(D, B, lamda, pca, error) 

            % Imports
            import nnf.alg.SRC;

            % For handling the occlusion or error with SRC
            if (error)
                dim1 = size(D, 1);
                D = [D eye(dim1)]; 
            end 

            [D, B] = SRC.pre_process(D, B, pca);
            
            te_n = size(B, 2);
            x = zeros(size(D,2), te_n);
            for i = 1 : te_n
                x(:, i) = SRC.lasso_ADMM(D, B(:,i), lamda, 1, 1);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ rate, label, dist_matrix ] = lass_recogn(D, B, ld, lb, coeff, p_norm, pca, error )

            % Imports
            import nnf.alg.SRC;
            
            if (error) % For handling the occlusion or error with SRC
                dim1 = size(D, 1);
                D = [D eye(dim1)]; 
            end

            [D, B] = SRC.pre_process(D, B, pca);

            d = size(coeff,2); % No. of test samples
            rate = zeros(1,d);
            label = zeros(1,d);
            idGroup = unique(ld(1,:)); idNum = length(idGroup);    
            dist_matrix = zeros(idNum, d); % Residual norm(..) for each class.

            for i = 1 : d
                x = coeff(:,i);        
                [temp, dist_matrix(:, i), ~] = SRC.src(D, B(:,i), ld(1,:), x, p_norm);        
                label(i) = temp(1);
                rate(i) = label(i) == lb(1,i);
            end
            rate = sum(rate)/d;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ ly, residuals,r ] = src(D, y, lrid, x, p)
        %  label = label(min(||y - R * xi||^2))
            % initiallization
            idGroup = unique(lrid);
            idNum = length(idGroup);
            residuals = zeros(idNum,1);

            % generate coefficients for each group
            xx = repmat(x,[1,idNum]);
            for i = 1 : idNum
                idx = lrid ~= idGroup(i);
                xx(idx, i) = 0;
            end

            % using the coefficients to calculate residuals
            r = repmat(y, [1, idNum]) - D*xx;
            for i = 1 : idNum
                residuals(i) = norm(r(:, i), p);
            end

            % find y's label that corresponding to the minimal residual
            ly = idGroup(min(residuals)==residuals);
            ly = ly(1);  
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

            % Imports
            import nnf.alg.SRC;
            
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
            [L U] = SRC.factor(A, rho);

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
                z = SRC.shrinkage(x_hat + u, lambda/rho);

                % u-update
                u = u + (x_hat - z);

                % diagnostics, reporting, termination checks
                history.objval(k)  = SRC.objective(A, b, lambda, x, z);

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
    end
    
end

