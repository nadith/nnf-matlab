classdef DCC < nnf.alg.MCC
    %DCC: Discriminant Co-entropy Criteria algorithm.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - DCC.l2 
    %   - DCC.test_l2 
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [W, U, dinfo] = l2(nndb_tr, nndb_tr_val, info) 
            % L2: leanrs the DCC subspace with l2-norm.
            %
            % Parameters
            % ----------
            % nndb_tr : nnf.db.NNdb
            %     Data object that contains the training database.
            %
            % nndb_tr_val : nnf.db.NNdb, optional
            %     Data object that contains the training validation database. (Default value = []).
            %
            % info : struct, optional
            %     Provide additional information to perform DCC. (Default value = []).    
            %
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.W                = rand(...);% Initial projection (i.e LDA projection, etc)
           	%     inf.grad.alpha       = 0.002;    % Projection 'W' learning rate
            %     inf.grad.alpha_mntm  = 0.001;    % Projection 'W' learning momemtum
            %     inf.grad.beta        = 0.3;      % Class mean learning rate 
            %     inf.grad.beta_mntm   = 0.5;      % Class mean learning momemtum
            %
            %     Note: beta rates must be higher than that of alpha rates for better stability.
            %
            % Returns
            % -------
            % W : 2D matrix -double
            %     Projecttion matrix.
            %
         	% U : 2D matrix -double
            %     New estimated class means.
            %
         	% dinfo : struct
            %     DCC info structure that has information for dcc_norm use. i.e dinfo.cls_sigma
            %
            %
            % Examples
            % --------
            % import nnf.alg.DCC;
            % [W, U] = DCC.l2(nndb_tr)
            % 
            % import nnf.alg.DCC;
            % [W, U] = DCC.l2(nndb_tr, nndb_tr_val);
            %
            % import nnf.alg.DCC;
            % info.W = DCC.l2(nndb_tr);
            % info.grad.alpha = 0.003;
            % inf.grad.alpha_mntm = 0.005;
            % [W, U] = DCC.l2(nndb_tr, [], info);
            %
            % import nnf.alg.DCC;
            % info = [];
            % info.g_norm = true;            
            % [W, U, dinfo] = DCC.l2(nndb_tr)
            % info.cls_sigma = dinfo.sigma;
            % accurary = Util.test(W, nndb_tr, nndb_te, info);  
            %
            
            % Set defaults for arguments
            if (nargin < 2), nndb_tr_val = []; end
            if (nargin < 3), info = []; end
            
            % Perform DCC optimization
            [W, U, dinfo] = optimize(nndb_tr, nndb_tr_val, info)
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [W, U, dinfo] = test_l2(info) 
            % TEST_L2: tests the DCC algorithm with a toy example.   
            %
            % Parameters
            % ----------
            % info : struct, optional
            %     Provide additional information to perform DCC. (Default value = []).    
            %
            %     Info Structure (with defaults)
            %     -----------------------------------
           	%     inf.test_1D  = true;  % 1D toy example
            %     inf.test_2D  = false; % 2D toy example
            %     inf.plot_LDA = false; % Plot LDA projection 
            %
            %
            % Returns
            % -------
            % W : 2D matrix -double
            %     Projecttion matrix.
            %
         	% U : 2D matrix -double
            %     New estimated class means.
            %
         	% dinfo : struct
            %     DCC info structure that has information for g_norm use. i.e dinfo.cls_sigma
            %
            %
            % Examples
            % --------
            % import nnf.alg.DCC;
            % [W, U] = DCC.l2(nndb_tr)
            %
            % import nnf.alg.DCC;
            % info = [];
            % info.g_norm = true;            
            % [W, U, dinfo] = DCC.l2(nndb_tr)
            % info.cls_sigma = dinfo.sigma;
            % accurary = Util.test(W, nndb_tr, nndb_te, info);
            %
            
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
            import nnf.alg.DCC;
            
            % Set defaults for arguments
            if (nargin < 1), info = []; end
            
            % Set defaults for info fields, if the field does not exist   
            if (~isfield(info,'test_1D')); info.test_1D = true; end;   
            if (~isfield(info,'test_2D')); info.test_2D = false; end; 
            if (~isfield(info,'plot_LDA')); info.plot_LDA = false; end; 
            
            % Initialize W, plot
            if (info.test_1D)
                %W = rand(2, 1);
                W = [1; 0];
                W = normc(W); % Unit vector normalization
                
                % Plot initial W
                plot(W(1, 1)*[-10 10], W(2, 1)*[-10 10], 'm'); hold on;
                
            elseif (info.test_2D)
                W = rand(2, 2);
                % W = [1 0; 0 1];                                               
                W = normc(W); % Unit vector normalization
            
                % Plot initial W
                plot(W(1, 1)*[-10 10], W(2, 1)*[-10 10], 'm'); hold on;
                plot(W(1, 2)*[-10 10], W(2, 2)*[-10 10], 'm'); hold on;
            end
                          
            % Sample 2D dataset for 2 classes
            X = [1.98 1.98; 2 2; 2.02 2.02; 14 14; 5 4.98; 5 5; 5 5.02; 5.5 5.02]';
            XL = [1 1 1 1 2 2 2 2];
            
            % Craete a NNdb object
            nndb = NNdb('Toy Dataset', X, 4, false, XL, Format.H_N);
            clear X;
            clear XL;          

            % Featch features for class individually 
            X1 = nndb.get_features(1);
            X2 = nndb.get_features(2);
            
            figure;
            
            % Plot the data
            plot(X1(1, :), X1(2, :), 'g*'); hold on;
            plot(X2(1, :), X2(2, :), 'b*'); hold on;

            % LDA projection plot
            if (info.plot_LDA)
                import nnf.alg.LDA;
                W_LDA = LDA.fl2(nndb);
                plot(W_LDA(1, 1)*[-10 10], W_LDA(2, 1)*[-10 10], 'r'); hold on;
            end                      
            
            % Perform DCC optimization
            %info.W = W;
            info.plot = true;
            [W, U, dinfo] = DCC.optimize(nndb, [], info);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    end
    
    methods (Access = private, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [W, U, info] = optimize(nndb_tr, nndb_val, info) 
            % OPTIMIZE: finds an optimal solution for DCC objective.
            %
            % Parameters
            % ----------
            % info : struct, optional
            %     Provide additional information to perform DCC opt. (Default value = []).    
            %
            %     Info Structure (with defaults)
            %     -----------------------------------
           	%     inf.test_1D           = true;     % 1D toy example
            %     inf.test_2D           = false;    % 2D toy example
            %     inf.plot              = false;    % Plot projections and samples
            %     inf.W                 = rand(...);% Initial projection (i.e LDA projection, etc)
           	%     inf.grad.alpha        = 0.002;    % Projection 'W' learning rate
            %     inf.grad.alpha_mntm   = 0.001;    % Projection 'W' learning momemtum
            %     inf.grad.beta         = 0.3;      % Class mean learning rate 
            %     inf.grad.beta_mntm    = 0.5;      % Class mean learning momemtum
            %
            %     Note: beta rates must be higher than that of alpha rates for better stability.
            %
            
            % Imports
            import nnf.alg.DCC;
            import nnf.alg.Util;
                        
            % Set defaults for arguments
            if (nargin < 2), nndb_val = []; end
            if (nargin < 3), info = []; end
            
            % Set defaults for info fields, if the field does not exist   
            if (~isfield(info,'W')); info.W = []; end; 
            if (~isfield(info,'plot')); info.plot = false; end;     
            if (~isfield(info,'grad')); info.grad = []; end;            
            if (~isfield(info.grad,'alpha')); info.grad.alpha = 0.002; end;
            if (~isfield(info.grad,'alpha_mntm')); info.grad.alpha_mntm = 0.001; end;            
            if (~isfield(info.grad,'beta')); info.grad.beta = 0.3; end;
            if (~isfield(info.grad,'beta_mntm')); info.grad.beta_mntm = 0.5; end;
                  
            % Set working varibles
            X = nndb_tr.features;
            n_per_class = double(unique(nndb_tr.n_per_class));
            n = nndb_tr.n;
            cls_n = nndb_tr.cls_n;
            d = nndb_tr.h * nndb_tr.w;
                                   
            % Error handling
            if (numel(n_per_class) > 1)
                error(['DCC_OPT_ERR: Multiple samples per class (n_per_class) is not supported.']);
            end
            
         	% Initialize W
            if (~isempty(info.W))
                W = info.W; 
            else
                W = orth(rand(nndb_tr.h, cls_n-1));
                W = normc(W); % Unit vector normalization
            end             
            
            % [PERF] Summation Matrix for efficient calculation of class mean, class mean gradient.
            sum_MAT = zeros(n, cls_n);
            vec = uint8([ones(n_per_class, 1); zeros(n - n_per_class, 1)]);        
            for i=[1:cls_n]
                sum_MAT(:, i) = vec;
                vec = circshift(vec, n_per_class);                   
            end
            
            % i.e For 2 class database
            %
            %     sum_MAT = [ 1 0;
            %                 1 0;
            %                 1 0;
            %                 1 0;
            %                 0 1;
            %                 0 1;
            %                 0 1;
            %                 0 1 ]
            
            % Calculate class mean, total mean
            U = repmat((X * sum_MAT)/n_per_class, n_per_class, 1);
            U = reshape(U(:), d, []); % dimension (d x n), same as X
            U0 = repmat(mean(U, 2), 1, n);
            
            % [IMPORTANT] beta rate should be higher than alpha since data means need to be learned (beta)
            % in a higher rate than the weight learning rate (alpha)
            beta_vU = zeros(size(U));
            beta_mntm = info.grad.beta_mntm;
            beta = info.grad.beta;
            
            alpha_vW = zeros(size(W));
        	alpha_mntm = info.grad.alpha_mntm;
            alpha = info.grad.alpha;

            % [PERF] Momentum switches, currently not occupied
            % mom_switch_iter_a = 100;
            % mom_switch_iter_b = 3000;
            
            % Gains
            a_gains = ones(size(W));
            b_gains = ones(size(U)); 
            
            % Iteration Count
            iter_c = 1;
            
            % Cost of previous iteration
            prev_cost = 0;
            
            % Tolerance count for convergence
            tol_c = 0;
            
            % Optimal W, U
            W_OPT = W;   
            U_OPT = U; 
            
            % Set cost information structure needed for cost evaluation
            ci.W = W;
            ci.X = X;            
            ci.U = U;
            ci.U0 = U0;
            ci.n_per_class = nndb_tr.n_per_class;
            ci.cls_n = cls_n;
            ci.n = n;
                
            % Plot handles
            h = []; h1 = []; h2 = [];
            
            while true                
                %-----------------------------------------------------------------------------------        
                % Boost with momentun switches
                % if iter_c == mom_switch_iter_a
                %     alpha = 0.0001;        
                %     alpha_mntm = 0.7;
                % 
                %     beta = 1.2;
                %     beta_mntm = 0.8;
                % end
                % 
                % if iter_c == mom_switch_iter_b
                %     beta = 0.0001;
                %     beta_mntm = 0.8;
                % end         

                %-----------------------------------------------------------------------------------        
                % Evaluate cost and request dU gradient
                cd1 = DCC.eval_cost(ci, true, false);

                % Gradient for each sample in U (need duplication)
                dU = cd1.dU * sum_MAT;
                dU = repmat(dU, n_per_class, 1);                 
                dU = reshape(dU(:), d, []); % dimension (d x n), same as X               
                
                % [PERF] Momenum and gain operations
                dU = beta * dU; 
                if(beta_mntm > 0)
                    b_gains = (b_gains + .3) .* (sign(dU) == sign(beta_vU)) ...       
                          + (b_gains * .7) .* (sign(dU) ~= sign(beta_vU));
                    b_gains(b_gains < 0.01) = 0.01;

                    beta_vU = beta_mntm * beta_vU + (b_gains.*dU);
                    dU = beta_vU;
                end  
                
                % Calculate new U
                U = U - dU;   
                
                % Calculate new U0
                U0 = [repmat(mean(U, 2), 1, n)];

                % Update the cost information structure 
                ci.U = U;
                ci.U0 = U0;
            
                %-----------------------------------------------------------------------------------        
                % Evaluate cost and request dW gradient
                cd2 = DCC.eval_cost(ci, false, true);
                
                % Print the cost in every 100 iteration
                if (mod(iter_c, 10) == 0)
                    disp(['cost1: ' num2str(cd1.err) ' cost2: ' num2str(cd2.err)]);                    
                end
                
                % Evaluate the accuracy every 100 iteration
                if (mod(iter_c, 100) == 0 && ~isempty(nndb_val))
                    accuracy = Util(W, NNdb('Temp db', X, n_per_class, true), nndb_val);  
                    disp(['ACC: ' num2str(accuracy)]);
                end

                % [PERF] Momenum and gain operations
                delta_W   = alpha * cd2.dW; 
                if(alpha_mntm > 0)
                    a_gains = (a_gains + .2) .* (sign(delta_W) == sign(alpha_vW)) ...       
                          + (a_gains * .8) .* (sign(delta_W) ~= sign(alpha_vW));
                    a_gains(a_gains < 0.01) = 0.01;

                    alpha_vW = alpha_mntm * alpha_vW + (a_gains.*delta_W);
                    delta_W = alpha_vW;
                end  
                
                % Calculate new W
                W = W - delta_W; 
                
                % Normalize W to unit vectors
                % Note: Orthogonality constraint now have this integrated
                % W = normc(W); 
                
                % Update the cost information structure
                ci.W = W;
                
                % Convergence check
                % Note: Converge if error stabilizes for more than the 100
                cd3 = DCC.eval_cost(ci, false, false);
                info.cls_sigma = cd3.cls_sigma;
                if (abs(prev_cost - cd3.err) < 1e-5)
                    if (tol_c > 100)
                        break;            
                    else
                        tol_c = tol_c + 1;
                    end
                else
                    tol_c = 0;
                    W_OPT = W;  
                    U_OPT = U;
                end
                
                % Update
                prev_cost = cd3.err;
                iter_c    = iter_c + 1;
                
                % Plot requirement
                if (~info.plot); continue; end;
                
                % Plot related
                delete(h);
                h = plot(U(1, :), U(2, :), 'r.', 'MarkerSize', 10); hold on;

                if (info.test_1D)
                    delete(h1);
                    h1  = plot(W(1, 1)*[-10 10], W(2, 1)*[-10 10]); hold on;

                elseif (info.test_2D)
                    delete(h1);
                    h1  = plot(W(1, 1)*[-10 10], W(2, 1)*[-10 10]); hold on;
                    
                    delete(h2);
                    h2 = plot(W(1, 2)*[-10 10], W(2, 2)*[-10 10]); hold on;
                end
                
                pause(0.05);
            end
            
            W = W_OPT;   
            U = U_OPT;
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [cd] = eval_cost(ci, req_dU, req_dW) 
            % EVAL_COST:
            %
            % ci.X = X;
            % ci.U = U;
            % ci.U0 = U0;
            % ci.n_per_class = n_per_class;
            % ci.cls_n = cls_n;

            % cd.err
            % cd.dU
            % cd.dW
            % cd.cls_sigma
            
            % Imports
            import nnf.alg.MCC;
            import nnf.utl.*;
            
            % Set defaults for arguments
            if (nargin < 2), req_dU = false; end
            if (nargin < 3), req_dW = false; end
            
            % C x N = total sample count                    
            cn = ci.n;

            % One-time initialization
            % persistent sintraM;
            % persistent sinterM;

            % if (isempty(sintraM))
            %     sintraM = calc_cls_sigma(col_norm_sqrd(X-U), CN, im_per_class, 1); % toy examples => 0.25*eye(8);
            %     sinterM = calc_cls_sigma(col_norm_sqrd(U-U0), CN, CN, 1); % toy examples => 0.25*eye(8);
            % end
            
            % sigma_intraM = sintraM;
            % sigma_interM = sinterM;
            
            % Calculate sigma matrices in each iteration
            sigma_intraM = MCC.calc_sigma_dcc(col_norm_sqrd(ci.X - ci.U), cn, ci.n_per_class, ci.cls_n);
            sigma_interM = MCC.calc_sigma_dcc(col_norm_sqrd(ci.U - ci.U0), cn, cn, 1);
    
            % Debuging purpose
            % D_XU = col_norm_sqrd(X-U);
            % D_UU = col_norm_sqrd(U-U0);
            % D_XU = col_norm_sqrd(W'*(X-U));
            % D_UU = col_norm_sqrd(W'*(U-U0));
    
            [gintra] = MCC.gauss(col_norm_sqrd(ci.W'*(ci.X - ci.U))*sigma_intraM);
            [ginter] = MCC.gauss((1./col_norm_sqrd(ci.W'*(ci.U - ci.U0)))*sigma_interM);
                
            cd.cls_sigma = sigma_intraM;
            ci.lambda = 0.5;
            ci.lambda2 = 1;
   
            cd.err = (1/cn) * ( ...
                    (1-ci.lambda) * (1*cn - sum(gintra, 2)) + ...
                    ci.lambda * (1*cn - sum(ginter, 2)) ...
                    ) ...
                    + ci.lambda2 * sum(col_norm_sqrd(ci.W' * ci.W - eye(size(ci.W, 2))));
                
                
                
%                 gnorm_intra =
% 
%   Columns 1 through 7
% 
%     0.0436    0.0455    0.0474    0.0000    0.9946    0.9946    0.9946
% 
%   Column 8
% 
%     0.9529
% 
% 
% gnorm_inter =
% 
%    1.0e-55 *
% 
%   Columns 1 through 7
% 
%     0.2572    0.2572    0.2572    0.2572    0.2572    0.2572    0.2572
% 
%   Column 8
% 
%     0.2572
% 
% cost: 0.74542



             if (req_dU) 
                 
                % Calculate gradient dU for intra cost
                dU_intra = ((ci.W * ci.W') * (ci.U - ci.X) * diag(gintra)) * sigma_intraM; %/(sigma_intra^2);  

                % Calculate gradient dU for inter cost
                fact = (ginter./col_norm_sqrd(ci.W' * (ci.U - ci.U0)).^2);
                dU_inter = -((ci.W * ci.W') * (ci.U - ci.U0) * diag(fact)) * sigma_interM; % /(sigma_inter^2);   

                % Calculate gradient total dU
                cd.dU = (1/cn) * ((1-ci.lambda) * dU_intra + ci.lambda * dU_inter);        
                
                % Verify the gradient value with approximation
                %diff = GradCheckX(ci.W, cd.dU, ci.X, fh, ci.U, ci.U0, true);
                %diff = sum(diff, 2)    
             end
             
             if (req_dW)
                 
                % Calculate gradient dW for intra cost 
                dW_intra = ((ci.X - ci.U) * diag(gintra) * sigma_intraM * (ci.X - ci.U)' * ci.W); %/(sigma_intra^2);  

                % Calculate gradient dW for inter cost
                fact     = (ginter./col_norm_sqrd(ci.W' * (ci.U - ci.U0)).^2);
                dW_inter = -((ci.U - ci.U0) * diag(fact) * sigma_interM * (ci.U - ci.U0)' * ci.W); %/(sigma_inter^2);   

                % Calculate gradient dW for orthogonality constraint
                dW_I    = (4.*(ci.W' * ci.W -  eye(size(ci.W, 2)))*ci.W')';    

                % Calculate gradient total dW
                cd.dW = (1/cn) * ((1-ci.lambda)*dW_intra + ci.lambda*dW_inter) + ci.lambda2*dW_I;  
                
                % Verify the gradient value with approximation
                %diff = GradCheckW(W, delta_W, X, fh, U, U0);
                %diff = sum(diff, 2)     
             end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

