classdef SSRC
    %SSRC: Structured Sparse Coding algorithm and varients.
    %   Solves ||Y - A*P - B*Q - C*R||_F^2 + lamda_p * |P|_1 + lamda_q * |Q|_1 + lamda_r * |R|_1
    %   where A, B, C, ... are dictionaries and P, Q, R, ... are the corresponding coefficients.
    %
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - SSRC.l1
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [coeffs, nndb_reconst] = l1(nndb_Y, nndbs, info)
            % L1: Performs SSRC with l1-norm.
            %   Solves ||y - A*x1 - B*x2||_F^2 + lamda_p * |x1|_1 + lamda_q * |x2|_1
            %   where A, B, ... are dictionaries and x1, x2, ... are the corresponding coefficients.
            %   Ref: Robust face recognition via occlusion dictionary learning, Weihua Oua.
            %            
            % Parameters
            % ----------
            % nndb_Y : nnf.db.NNdb
            %     Target database.
            %
            % nndbs : `array_like` -nnf.db.NNdb
            %     Vector of dictionary databases.
            %
            % info : struct, optional
            %     Provide additional information to perform SSRC-L1. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.lambdas = [0.001 0.001];  % Lagrange multipliers for constraints: |x1|_1 <= epsilon, |x2|_1 <= epsilon
            %     inf.max_iter = 20;            % Maximum iterations to optimize the cost.
            %     inf.tolerance = 0.001;        % Tolerance level to optimize the cost.
            %     inf.visualize = false;        % Visualize the reconstructions.
            %
            % Returns
            % -------
            % accuracy : double
            %     Accuracy of classification.
         	%
            %
            % Examples
            % --------
            % import nnf.alg.SSRC;
            % nndbs = [nndb_dict1 nndb_dict2];
            % accuracy = SRC.l1(nndb_Y, nndbs)
            %        

            % Imports
            import nnf.alg.SRC;
            import nnf.db.NNdb;
            import nnf.db.Format;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 3), info = []; end
            
            % Set defaults for info fields, if the field does not exist  
            if (~isfield(info,'lambdas')); info.lambdas = 0.001 * ones(1, numel(nndbs)); end
            if (~isfield(info,'max_iter')); info.max_iter = 20; end 
            if (~isfield(info,'tolerance')); info.tolerance = 0.001; end
            if (~isfield(info,'visualize')); info.visualize = false; end
            
            % Initialization of coeffs cell array
            coeffs = cell(0, 0);
            n_per_class = unique(nndb_Y.n_per_class);
            for dbi=1:numel(nndbs)
                nndb = nndbs(dbi);
                coeffs{dbi} = double(zeros(nndb.n, nndb_Y.n));
            end

            prev_cost = [];
            iter = 1;
            while (iter <= info.max_iter)

                % Calculate cost
                cost = nndb_Y.features;
                cost_coeff = 0;
                for dbi=1:numel(nndbs)
                    cost = cost - nndbs(dbi).features * coeffs{dbi};
                    
                    coeff = coeffs{dbi};
                    for i=1:size(coeff, 2)
                        cost_coeff = cost_coeff + info.lambdas(dbi) * norm(coeff(:, i), 1);
                    end
                end
                cost = norm(cost)^2 + cost_coeff;
                disp(['COST: ' num2str(cost) ' ITER: ' num2str(iter)]);
                                
                if (~isempty(prev_cost) && (prev_cost - cost < info.tolerance))
                    disp(['cost:' num2str(cost) ' tolerance:' num2str(info.tolerance) ' breached.']);
                    break;
                end
                prev_cost = cost;
                
                % Step wise optimization for coeffs
                for dbj = 1:numel(nndbs)
                    Y = nndb_Y.features;

                    % Calculate (Y - nndbs(ndbi).features*P), where (ndbi ~= ndbj)
                    for dbi=1:numel(nndbs)
                        if (dbi == dbj); continue; end
                        Y = Y - nndbs(dbi).features * coeffs{dbi};
                    end

                    % Build NNdb object for database Y
                    nndb_P = NNdb('P', Y, n_per_class, true, [], Format.H_N);

                    % Update the coefficients for `nndbs(ndbj)` dictionary
                    info_src.lambda = info.lambdas(dbj);
                    info_src.mean_diff = false;
                    [~, ~, sinfo] = SRC.l1(nndbs(dbj), nndb_P, info_src);
                    coeffs{dbj} = sinfo.coeffs;
                end   

                iter = iter + 1;
            end
            if (iter >= info.max_iter); disp(['max_iter:' num2str(info.max_iter) ' reached.']); end
            
            DD = 0;
            for dbi=1:numel(nndbs)                        
                DD = DD + nndbs(dbi).features * coeffs{dbi};
            end
            DD = uint8(DD.*255);
            % DD = uint8(mapminmax(DD',0,255)'); % WARNING: Will have a high impact on classification accuracy
            nndb_reconst = NNdb('RECONST', reshape(DD, 30, 30, 1, []), nndb_Y.n_per_class, true);
            
            % Visualizations
            if (info.visualize)
                figure, nndb_reconst.show(10, 10)
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndb_Ad] = dict_learn_proj_err(nndb_Y, nndb_A0, info) 
            % dict_learn_proj_err: learns dictionary for projection error.
            %
            %   i.e, Assume y = A0 * p + Ad * q + e, Hence in ||Y - A0 * P  - Ad * Q||, vectors in Y
            %   are projected onto their corresponding true classes in A0 => Y_proj_Ac.
            %
            %   The dictionary `Ad` that represents P_error = (Y - Y_proj_Ac) is learnt 
            %   via the fomulation,
            %
            %   ||p - Ad*x||_2^2 + lambda1 * |x|_1 + lambda2 * ||A0' * Ad||_2^2
            %   s.t d_j' * d_j = 1, j = 1, 2, 3, ...
            %
            %   A0: dictionary to represent reconstructed image (= y - Ad * q),
            %   Ad: dictionary to represent project error. (= y - y_proj_A0, = y - A0 * p)
            % 
            %   Ref: Robust face recognition via occlusion dictionary learning, Weihua Oua.
            %
            % Parameters
            % ----------
            % nndb_Y : nnf.db.NNdb
            %     Target database, where y = A0 * p + Ad * q + e
            %
            % nndb_A0 : nnf.db.NNdb
            %     Dictionary database to reprsent targets on the subspace. (0-1 normalized)
            %
            % info : struct, optional
            %     Provide additional information to perform SRC. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.lambda1 = 0.02;    % Lagrange multiplier for constraint: |x|_1 <= epsilon
            %     inf.lambda2 = 0.5;     % Lagrange multiplier for constraint: A0' * Ad = 0 (mutual coherence)
            %     inf.gamma = 0.01;      % Lagrange multiplier for constraint: d_j' * d_j - 1 = 0
            %     inf.max_iter = 20;     % Maximum iterations to optimize the cost.
            %     inf.tolerance = 0.001; % Tolerance level to optimize the cost.
            %     inf.visualize = false; % Visualize projections and the learnt dictionary.
            %
            % Returns
            % -------
            % nndb_Ad : nnf.db.NNdb
            %     Learnt dictionary to represent projection error. (= y - y_proj_A0)
         	%
            % Notes
            % -----
            %   nndb_Y.cls_lbl needs to match with nndb_A0.cls_lbl in order to perform class
            %   specific projection. If not, the projection is done on a randomly chosen class
            %   subspace.
            %
            % Examples
            % --------
            % import nnf.alg.SSRC;
            % nndb_Ad = SSRC.dict_learn_P_error(nndb_Y, nndb_A0, nndb_Ad)
            % 
            
            % Imports
            import nnf.alg.SRC;
            import nnf.db.NNdb;
            import nnf.db.Format;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 3), info = []; end
            
            % Set defaults for info fields, if the field does not exist  
            if (~isfield(info,'lambda1')); info.lambda1 = 0.02; end
            if (~isfield(info,'lambda2')); info.lambda2 = 0.5; end
            if (~isfield(info,'gamma')); info.gamma = 0.01; end
            if (~isfield(info,'max_iter')); info.max_iter = 20; end 
            if (~isfield(info,'tolerance')); info.tolerance = 0.001; end
            if (~isfield(info,'visualize')); info.visualize = false; end
            
            P = zeros(nndb_Y.h * nndb_Y.w * nndb_Y.ch, nndb_Y.n);
            st = 1;
            for i=1:nndb_Y.cls_n   
                Y_cls_st = nndb_Y.cls_st(i);    
                Y_n_per_class = uint32(nndb_Y.n_per_class(i));
                Y_cls_lbl = nndb_Y.cls_lbl(Y_cls_st);                
                
                Y = nndb_Y.get_features(Y_cls_lbl);
                A = nndb_A0.get_features(Y_cls_lbl, 'l2');

                if (isempty(A))
                    % randomly choose Y_cls_lbl 
                    A0_cls_i = randperm(nndb_A0.cls_n, 1);
                    Y_cls_lbl = nndb_A0.cls_lbl(nndb_A0.cls_st(A0_cls_i));        
                    A = nndb_A0.get_features(Y_cls_lbl, 'l2');
                end

                % Calculate projection error
                P(:, st:st + Y_n_per_class - 1) = (Y - (A/(A'*A))*(A'*Y));
                st = st + Y_n_per_class;
            end

            % Build NNdb object for projections
            nndb_P = NNdb('', P, nndb_Y.n_per_class, true, [], Format.H_N);
     
            % Iterative optimization for coeff, `Ad` dictionary
            A0 = nndb_A0.get_features([], 'l2'); % Normalize
            Ad = rand(nndb_P.h*nndb_P.w*nndb_P.ch, 50);
            Ad = normc(Ad); % Normalize
                        
            prev_cost = [];
            iter = 1;
            while (iter < info.max_iter)                
                % Build NNdb object for database Ad
                nndb_Ad = NNdb('Ad', Ad, 1, true, [], Format.H_N);

                info_src.lambda = info.lambda1;
                info_src.mean_diff = false;
                info_src.only_coeff = true;
                %info_src.method.name = 'SRV1_9.SRC2.interiorPoint'; % Better results
                %info_src.method.name = 'L1BENCHMARK.FISTA'; % or default-ADMM;
                %info_src.method.param.tolerance = 0.0001;
                [~, ~, sinfo] = SRC.l1(nndb_Ad, nndb_P, info_src);
                coeffs = sinfo.coeffs;
                
                cost = norm(P - Ad * coeffs)^2 + info.lambda1 * norm(coeffs, 1) + ...
                        info.lambda2 * norm(A0' * Ad)^2;
                disp(['COST: ' num2str(cost) ' ITER: ' num2str(iter)]);
                    
                if (~isempty(prev_cost) && (prev_cost - cost < info.tolerance))
                    disp(['cost:' num2str(cost) ' tolerance:' num2str(info.tolerance) ' breached.']);
                    break;
                end
                prev_cost = cost;
                
                for l=1:nndb_Ad.n
                    Bl = coeffs(l, :);
                    Z = P - (Ad * coeffs - Ad(:, l) * Bl);
                    dl = ((Bl*Bl' - info.gamma) * eye(size(A0, 1)) + ...
                            info.lambda2 * (A0 * A0')) \ (Z * Bl');                    
                    Ad(:, l) = dl/norm(dl);
                end
                
                iter = iter + 1;
            end
            if (iter >= info.max_iter); disp(['max_iter:' num2str(info.max_iter) ' reached.']); end
            
            % Visualizations
            if (info.visualize)
                DD = P.*255;
                DD = uint8(mapminmax(DD',0,255)');
                figure, immap(reshape(DD, 30, 30, 1, []), 10, 10, 'Title', 'Projection Errors');
                
                DD = Ad.*255;
                DD = uint8(mapminmax(DD',0,255)');
                figure, immap(reshape(DD, 30, 30, 1, []), 10, 10, 'Title', 'Dictionary Ad');
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

