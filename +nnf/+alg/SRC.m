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
        function [accuracy, dist, sparsity, coeff, time] = l1(nndb_g, nndb_p, info) 
            % L1: performs SRC with l1-norm.
            % TODO: return sparse coefficient and perform the recognition in Util class
            %
            % Parameters
            % ----------
            % nndb_g : nnf.db.NNdb
            %     Dictionary database.
            %
            % nndb_p : nnf.db.NNdb
            %     Target database.
            %
            % info : struct, optional
            %     Provide additional information to perform SRC. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.dist = false;                 % Calculate distance matrix. TODO: implement
            %     inf.noise = false;                % Consider error/noise representation with identity matrix.
            %     inf.mean_diff = true;             % Use feature mean differences.
            %     inf.lambda = 0.01;                % Lagrange multiplier for constraint: |x|_1 <= epsilon
            %     inf.only_coeff = false;           % Learn only the sparse coefficients. No classification.
            %     inf.method.name = 'DEFAULT.SRC';  % Optimization method. (Default - ADMM).
            %                                       % i.e   'SRV1_9.SRC1'
            %                                               'SRV1_9.SRC2.interiorPoint', 
            %                                               'SRV1_9.SRC2.activeSet', 'SRV1_9.SRC2.proximal', 'SRV1_9.SRC2.smo'
            %                                               'L1BENCHMARK.L1LS'
            %                                               'L1BENCHMARK.FISTA'
            %     inf.method.param:             % Optimization method specific parameters described below
            %
            %     `inf.method.param` Structure (with defaults)
            %     --------------------------------------------
            %     method: 'SRV1_9.SRC1'
            %     param.epsilon = 0.05;         % The tolerance in the stable version.
            %     param.randomCorrupt = false;  % Use the version for corrution/noise
            %     param.rubost = false;         % run the stable/robust version of SCR1.
            %     param.predicter = 'subspace'; % Rule to interpret the sparse code. {'subspace','max','kvote'}.
            %
            %     method: 'SRV1_9.SRC2.interiorPoint'
            %     param.predicter = 'subspace';     % Rule to interpret the sparse code. {'subspace', 'max', 'knn'}.
            %     param.knn = numel(nndb_g.cls_lbl);% No of nearest neighbours to be used if predicter == 'knn'
            %
            %     method: 'SRV1_9.SRC2.activeSet', 'SRV1_9.SRC2.proximal', 'SRV1_9.SRC2.smo'
            %     param.predicter = 'subspace';     % Rule to interpret the sparse code. {'subspace', 'max', 'knn'}.
            %     param.knn = numel(nndb_g.cls_lbl);% No of nearest neighbours to be used if predicter == 'knn'
            %     param.kernel = 'linear';          % Kernel to compute kernal matrix. {'linear','polynomial','rbf','sigmoid','ds'}
            %     param.param = 0;                  % Kernel parameter.
            %
            %     method: 'L1BENCHMARK.FISTA', 'L1BENCHMARK.L1LS'
            %     param.tolerance = 1e-04;          % tolerance for stopping criterion. 
            %                                       % Note: affects accuracy and time to converge.
            %     
            %
            % Returns
            % -------
            % accuracy : double
            %     Accuracy of classification.
            %
            % distance : double
            %     Distance matrix. (TEST_SAMPLES x GALLERY_SAMPLES)
            %
            % sparsity : double
            %     Scalar indicating the sparsity.
            % 
            % coeff : `array_like` -double
            %     Sparse coefficient vector.
            % 
            % time : double
            %     Consumed time in seconds in optimization.
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
            if (~isfield(info,'dist')); info.dist = false; end
            if (~isfield(info,'noise')); info.noise = false; end
            if (~isfield(info,'mean_diff')); info.mean_diff = false; end
            if (~isfield(info,'lambda')); info.lambda = 0.01; end
            if (~isfield(info,'only_coeff')); info.only_coeff = false; end
            if (~isfield(info,'method')); info.method = struct; end
            if (~isfield(info.method,'name')); info.method.name = 'DEFAULT.SRC'; end
            if (~isfield(info.method,'param')); info.method.param = []; end
            
            % Initialize output params
            accuracy = [];
            dist = [];
            sparsity = [];

            A = nndb_g.features;
            Y = nndb_p.features;
            
            if (info.noise)
                % Augment identity matrix to represent the error sparsely.
                % Assumption: RSRC assumes error/occlusion/etc are sparse on identity matrix
                A = [A eye(size(A, 1))];
            end
            
            if (info.mean_diff)
                m = mean(A, 2);
                A = A - repmat(m, [1,size(A, 2)]);
                Y = Y - repmat(m, [1,size(Y, 2)]);
            end
            
            % Normalize columns of matrices
            A = normc(A);
            Y = normc(Y);
                     
            % TODO:
            %
            % Model selection (with different lambda)
            % [param,maxAcc]=lineSearchSRC2(trainSet,trainClass);
            % option.lambda=param;
            %
            
            switch (info.method.name)
                
                case 'SRV1_9.SRC1'
                    options = info.method.param;
                    if (~isfield(options,'epsilon')); options.epsilon = 0.05; end
                    if (~isfield(options,'randomCorrupt')); options.randomCorrupt = false; end
                    if (~isfield(options,'rubost')); options.rubost = false; end
                    
                    % 'subspace', 'max', 'kvote'
                    if (~isfield(options,'predicter')); options.predicter = 'subspace'; end
                    
                    % If only coefficents are required, do not perform classification
                    if (info.only_coeff); options.predicter = ''; end

                    % TODO: Troubleshoot POSTIVE DEFNITE error
                    [testClassPredicted, sparsity, dist, coeff, time] = SRC.src__(A, nndb_g.cls_lbl', Y, options);
                                        
                case {'SRV1_9.SRC2.interiorPoint', 'SRV1_9.SRC2.activeSet', 'SRV1_9.SRC2.proximal', 'SRV1_9.SRC2.smo'}                    
                    options = info.method.param;
                    if (~isfield(options,'lambda')); options.lambda = info.lambda; end
                    if (~isfield(options,'method')); idxs = strfind(info.method.name, '.'); options.method = info.method.name(idxs(end)+1:end); end
                               
                    % 'subspace', 'max', 'knn'
                    if (~isfield(options,'predicter')); options.predicter = 'subspace'; end 
                    if (strcmp(options.predicter, 'knn') && ~isfield(options,'knn')); options.knn = numel(nndb_g.cls_lbl); end % no of nearest neighbours
                    
                    % If only coefficents are required, do not perform classification
                    if (info.only_coeff); options.predicter = ''; end
                    
                    if (strcmp(options.method, 'activeSet') || ...
                            strcmp(options.method, 'proximal') || ...
                            strcmp(options.method, 'smo'))
                        
                        % 'rbf', 'polynomial', 'linear', sigmoid, 'ds' % dynamical systems kernel
                        if (~isfield(options,'kernel')); options.kernel = 'linear'; end
                        if (~isfield(options,'param')); options.param = 0; end
                    end
                          
                    % Assumption: Normalized before applying SRC algorihm
                    [testClassPredicted, sparsity, dist, coeff, time] = SRC.SRC2__(A, nndb_g.cls_lbl', Y, options);
                                        
                case {'L1BENCHMARK.FISTA', 'L1BENCHMARK.L1LS'} 
                    options = info.method.param;
                    if (~isfield(options,'lambda')); options.lambda = info.lambda; end
                    if (~isfield(options,'tolerance')); options.tolerance = 1e-04; end
                    
                    coeff = zeros(size(A, 2), size(Y, 2));                    
                    tic
                    for i=1:size(Y, 2)
                        if (strcmp(info.method.name, 'L1BENCHMARK.FISTA'))
                            [coeff(:, i)] = SolveFISTA(A, Y(:, i), 'lambda', options.lambda, 'tolerance', options.tolerance);
                            
                        else % 'L1BENCHMARK.L1LS'
                            [coeff(:, i)] = SolveL1LS(A, Y(:, i), 'lambda', options.lambda, 'tolerance', options.tolerance);
                        end
                    end
                    time = toc;
                    sparsity = sum(sum(abs(coeff)<=0.0001))/(size(coeff, 1) * size(coeff, 2));
                    
                case {'DEFAULT.SRC'}
                    tic
                    coeff = SRC.train_lasso__(A, Y, info.lambda);
                    time = toc;
                    sparsity = sum(sum(abs(coeff)<=0.0001))/(size(coeff, 1) * size(coeff, 2));
                    
                otherwise
                    error(['Invalid `info.method.name`: ' info.method.name]);
            end

            % If only coefficents are required, return from this point.
            if (info.only_coeff); return; end
                        
            % Perform classification and/or calculate accuracy, distance matrix
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            idxs = strfind(info.method.name, '.');
            
            % Sparse Representation Toolbox in MATLAB, int_lib: SRV1_9
            if (strcmp(info.method.name(1:idxs(1)), 'SRV1_9.'))
                accuracy = (sum(testClassPredicted == nndb_p.cls_lbl')/nndb_p.n)* 100;
                                
            else % L1-BENCHMARK Framework or DEFAULT.SRC - ADMM
                [accuracy, ~, dist] = SRC.lass_recogn__(A, Y, nndb_g.cls_lbl, nndb_p.cls_lbl, coeff, 2);                
            end
            
            % Calculate distance matrix
            if (info.dist)
                eye_ex = eye(nndb_g.cls_n);
                eye_ex = repmat(eye_ex, 1, double(unique(nndb_g.n_per_class)))';
                eye_ex = reshape(eye_ex, nndb_g.cls_n, []);
                dist = dist * eye_ex;
            else
                dist = [];
            end
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function SRC_MCC()
            % TODO: SLINDA
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ rate, label, dist_matrix ] = lass_recogn(A, Y, lbl_A, lbl_B, coeff, p_norm, pca )

            % Imports
            import nnf.alg.SRC;
            
            % if (error) % For handling the occlusion or error with SRC
            %     dim1 = size(A, 1);
            %     A = [A eye(dim1)]; 
            % end
            % 
            % [A, Y] = SRC.pre_process(A, Y, pca);

            d = size(coeff,2); % No. of test samples
            rate = zeros(1,d);
            label = zeros(1,d);
            idGroup = unique(lbl_A(1,:)); idNum = length(idGroup);    
            dist_matrix = zeros(idNum, d); % Residual norm(..) for each class.

            for i = 1 : d
                x = coeff(:,i);        
                [temp, dist_matrix(:, i), ~] = SRC.src_pred__(A, Y(:,i), lbl_A(1,:), x, p_norm);        
                label(i) = temp(1);
                rate(i) = label(i) == lbl_B(1,i);
            end
            rate = sum(rate)/d;
            dist_matrix = dist_matrix';
        end
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private, Static) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % IMPLEMENTATION SRC_DEFAULT - ADMM(FROM XIN):
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
        function [A, Y] = pre_process(A, Y, pca)
            % TODO: Remove this method and incorporate it to NNdb class
            [~,d1] = size(D);
            [~,d2] = size(Y);

            % Calculate the diff (offset from mean)
            m = mean(D, 2);
            D = D - repmat(m, [1,d1]);
            Y = Y - repmat(m, [1,d2]);

            % Imports
            import nnf.alg.SRC;
            
            % Normalize columns of matrices
            A = normc(A);
            Y = normc(Y);

            if (pca > 0)
                v = zPCA(A, pca); %? missing
                A = v'*A;
                Y = v'*Y;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function x = train_lasso__(A, Y, lamda, pca) 
            % minimize ||y - A*x||^2 + lambda * |x|_1,            
            
            % Imports
            import nnf.alg.SRC;

%             % For handling the occlusion or error with SRC
%             if (error)
%                 dim1 = size(A, 1);
%                 A = [A eye(dim1)]; 
%             end 
% 
%             [A, Y] = SRC.pre_process(A, Y, pca);
            
            te_n = size(Y, 2);
            x = zeros(size(A,2), te_n);
            for i = 1 : te_n
                x(:, i) = SRC.lasso_ADMM__(A, Y(:,i), lamda, 1, 1);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ rate, label, dist_matrix ] = lass_recogn__(A, Y, lbl_A, lbl_B, coeff, p_norm, pca )

            % Imports
            import nnf.alg.SRC;
            
            % if (error) % For handling the occlusion or error with SRC
            %     dim1 = size(A, 1);
            %     A = [A eye(dim1)]; 
            % end
            % 
            % [A, Y] = SRC.pre_process(A, Y, pca);

            d = size(coeff,2); % No. of test samples
            rate = zeros(1,d);
            label = zeros(1,d);
            idGroup = unique(lbl_A(1,:)); idNum = length(idGroup);    
            dist_matrix = zeros(idNum, d); % Residual norm(..) for each class.

            for i = 1 : d
                x = coeff(:,i);        
                [temp, dist_matrix(:, i), ~] = SRC.src_pred__(A, Y(:,i), lbl_A(1,:), x, p_norm);        
                label(i) = temp(1);
                rate(i) = label(i) == lbl_B(1,i);
            end
            rate = sum(rate)/d;
            dist_matrix = dist_matrix';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ ly, residuals, r] = src_pred__(D, y, lrid, x, p)
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
        function [z, history] = lasso_ADMM__(A, b, lambda, rho, alpha) 
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
        % IMPLEMENTATION SRV1_9.SRC1 - Sparse Representation Toolbox in MATLAB, int_lib: SRV1_9
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [testClassPredicted,sparsity,otherOutput,coeff,time] = src__(trainSet,trainClass,testSet,option)
            % sparse representation classification approach (SRC1)
            % trainSet, matrix, each column is a training sample
            % trainClass: column vector, the class labels for the training samples
            % testSet: matrix, each column is a new or testing sample
            % option, struct, with fields:
            % option.epsilon, scalar, the tolerance in the stable version, the default is 0.05.
            % option.predicter, string, the rule to interpret the sparse code, it can be
            % 'subspace' (default),'max','kvote'.
            % option.randomCorrupt, logical, if use the version for corrution/noise,
            % the default is false.
            % option.rubost, if run the stable/robust version of SCR1, the default is
            % false.
            % Yifeng Li
            % August 04, 2010
            % note: each sample has to be normalized to unit l2 norm before runing it

            % % normalization to length 1
            % trainSet=normc(trainSet);
            % testSet=normc(testSet);

            %optionDefault.p=16;
            optionDefault.epsilon=0.05;
            % optionDefault.predicter='subspace';
            optionDefault.predicter=''; % NADITH
            optionDefault.randomCorrupt=false;
            optionDefault.rubost=false;
            if nargin<4
               option=[]; 
            end
            option=mergeOption(option,optionDefault);
            % trainSet=downsample(trainSet,option.p);
            % testSet=downsample(testSet,option.p);

            if option.randomCorrupt
                trainSet=[trainSet,eye(size(trainSet,1),size(trainSet,1))];
            end

            % training step, obtain sparse coefficients in columns of Y
            Y=zeros(size(trainSet,2),size(testSet,2));
            % if option.randomCorrupt
            %     Y= trainSet\testSet;
            %     testSet=testSet-Y(end-size(trainSet,1)+1:end,:);
            %     Y=Y(1:size(trainSet,1),:);
            %     trainSet=trainSet(:,1:(size(trainSet,2)-size(trainSet,1)));
            % else
            %     for i=1:size(testSet,2)
            %         y0=pinv(trainSet)*testSet(:,i);
            %         Y(:,i)= l1qc_logbarrier(y0, trainSet, [], testSet(:,i), option.epsilon);
            %     end
            % end

            tic
            for i=1:size(testSet,2)
                if option.randomCorrupt
                    y0=pinv(trainSet)*testSet(:,i);
                    yi=l1eq_pd(y0, trainSet, [], testSet(:,i),option.epsilon);
                    Y(:,i)=yi;
                    testSet(:,i)=testSet(:,i)-yi(end-size(trainSet,1)+1:end);
                else
                    y0=pinv(trainSet)*testSet(:,i);
                    if option.rubost
                        Y(:,i)= l1qc_logbarrier(y0, trainSet, [], testSet(:,i), option.epsilon);
                    else
                        yi=l1eq_pd(y0, trainSet, [], testSet(:,i),option.epsilon);
                        Y(:,i)=yi;
                    end
                end
            end
            time = toc; % NADITH

            if option.randomCorrupt
                Y=Y(1:size(trainSet,2)-size(trainSet,1),:);
                trainSet=trainSet(:,1:(size(trainSet,2)-size(trainSet,1)));
            end
            % calculate sparsity
            sparsity=sum(sum(abs(Y)<=0.0001))/(size(Y,1)*size(Y,2));

            % NADITH
            otherOutput=[];
            testClassPredicted=[];
            
            % predict step
            switch option.predicter
                case  'max'
                    [val,ind]=max(Y,[],1);
                    testClassPredicted=trainClass(ind);
                case 'kvote'
                    for s=1:size(Y,2)
                        [sortedCoeff,ind] = getBestScores(Y(:,s),option.k);
                        predicted(s,:)=trainClass(ind);
                    end
                    testClassPredicted=vote(predicted);
                case 'subspace'
                    [testClassPredicted,residuals]=subspace(Y,testSet,trainSet,trainClass);
                    otherOutput=residuals;
            end
            
            coeff = Y; % NADITH
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % IMPLEMENTATION SRV1_9.SRC2 - Sparse Representation Toolbox in MATLAB, int_lib: SRV1_9
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [testClassPredicted, sparsity, otherOuptput, coeff, time] = SRC2__(trainSet, trainClass, testSet, option)
            % SRC2 for overcomplete data
            % trainSet: matrix, each column is a training sample
            % trainClass: column vector, the class labels for training samples
            % testSet: matrix, the test samples
            % option, struct, with fields:
            % option.lambda: scalar, the parameter to optimization algorithm l1_ls, the
            % default is 0.1.
            % option.predicter: string, the rule to interpret the sparse code, it can be
            % 'subspace' (default),'max','kvote'.
            % option.method: string, the optimization method. It can be 'activeSet',
            % 'interiorPoint', 'proximal', and 'smo'.
            % testClassPredicted: column vectors, the predicted class labels of
            % testing samples.
            % sparsity: scalar, the sparsity of the sparse coefficient matrix.
            % Yifeng Li
            % note: each sample has to be normalized to unit l2 norm

            % % normalization to length 1
            % trainSet=normc(trainSet);
            % testSet=normc(testSet);

            % optionDefault.p=4;
            optionDefault.lambda=0.1;
            optionDefault.kernel='linear';
            optionDefault.param=0;
            % optionDefault.predicter='subspace';
            optionDefault.predicter=''; % NADITH
            optionDefault.knn=numel(trainClass);
            optionDefault.method='activeSet';
            option=mergeOption(option,optionDefault);
            % trainSet=downsample(trainSet,option.p);
            % testSet=downsample(testSet,option.p);

            tic
            % training step, obtain sparse coefficients in columns of Y
            Y=zeros(size(trainSet,2),size(testSet,2));
            switch option.method
                case 'activeSet'
                    % active set algorithm
                    H=computeKernelMatrix(trainSet,trainSet,option);
                    C=computeKernelMatrix(trainSet,testSet,option);
                    [Y,~]=l1QPActiveSet(H,-C,option.lambda);
                case 'interiorPoint'
                    % interior point-method
            %         AtA=computeKernelMatrix(trainSet,trainSet,option);
            %         AtB=computeKernelMatrix(trainSet,testSet,option);
            %         BtB=computeKernelMatrix(testSet,testSet,option);
            %         BtB=diag(BtB);
            %         Y=l1LSKernelBatchDL(AtA,AtB,BtB,option);
                    for i=1:size(testSet,2)                        
                        Y(:,i)= l1_ls(trainSet, testSet(:,i), option.lambda); %http://www.stanford.edu/~body/l1_ls
                    end
                case 'proximal'
                    % proximal method
                    AtA=computeKernelMatrix(trainSet,trainSet,option);
                    AtB=computeKernelMatrix(trainSet,testSet,option);
                    BtB=computeKernelMatrix(testSet,testSet,option);
                    BtB=diag(BtB);
                    Y=l1LSProximal(AtA,AtB,BtB,option);
                case 'smo'
                    H=computeKernelMatrix(trainSet,trainSet,option);
                    C=computeKernelMatrix(trainSet,testSet,option);
                    Y=l1QPSMOMulti(H,-C,option.lambda);
                otherwise 
                    error('choose correct method for l1ls');
            end
            time = toc; % NADITH
            
            % calculate sparsity
            sparsity=sum(sum(abs(Y)<=0.0001))/(size(Y,1)*size(Y,2));

            % NADITH
            testClassPredicted = [];
            otherOuptput = [];
                    
            % predict step
            switch option.predicter
                case  'max'
                    [val,ind]=max(Y,[],1);
                    testClassPredicted=trainClass(ind);
                    otherOuptput=[];
                case 'knn'
                    testClassPredicted=knnrule(Y,trainClass,option.knn);
                    otherOuptput=[];
                case {'subspace','ns'}
                    [testClassPredicted,residuals]=subspace(Y,testSet,trainSet,trainClass);
                    otherOuptput=residuals;
            end
            
            coeff = Y; % NADITH
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    
end

