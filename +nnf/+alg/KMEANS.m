classdef KMEANS
    % KMEANS: K-means clustering algorithm
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - LDA.do (L2 distance)
    
    % Copyright @ Qilin Li, Curtin University (li.qilin@postgrad.curtin.edu.au).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [label, centers] = do(nndb, info)
            % do Kmeans clustering
            %
            % Parameters
            % ----------
            % nndb : nnf.db.NNdb
            %     Data object that contains the database.
            %
            % info : struct, optional
            %     Provide additional information to perform Kmeans. (Default value = []).
            %
            %
            % Returns
            % -------
            % label : nndb.n * 1 vector, indicating the labels for each
            %     sample
            %
            % centers : nndb.n * nndb.features vector. Each row is a
            %     centes of a class
            %
            % Examples
            % --------
            % import nnf.alg.KMEANS;
            % info.Replicates = 10;
            % info.MaxIter = 1000;
            % [label, centers] = KMEANS.do(nndb, info)
            %
            
            % Imports
            import nnf.libs.DengCai;
            import nnf.alg.KMEANS;
            % Set defaults for arguments
            if (nargin < 2), info = []; end
            if (~isfield(info,'MaxIter')); info.MaxIter = 50; end
            if (~isfield(info,'Replicates')); info.Replicates = 1; end
            
            % Perform kmeans
            [label, centers] = DengCai.litekmeans(nndb.features', nndb.cls_n, 'MaxIter', info.MaxIter, ...
                                                  'Replicates', info.Replicates);
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ffaces] = fisher_face_core(X, XL, info) 
            %FISHER_FACE_CORE: performs fisher face learning.
            
            % Imports
            import nnf.libs.DengCai;
            
            % Set defaults for info fields, if the field does not exist  
            if (~isfield(info,'Regu')); info.Regu = false; end; 
            if (~isfield(info,'Fisherface')); info.Fisherface = true; end; 
            
            % Set defaults for info fields, depending on LDA type
            if (info.Regu)
                
                % Set defaults for info fields, if the field does not exist
                if (~isfield(info,'ReguAlpha')); info.ReguAlpha = 0.1; end; 
                
                options.Regu = info.Regu;
                options.ReguAlpha = info.ReguAlpha;
                
            elseif (info.Fisherface)
                
                % Set defaults for info fields, if the field does not exist
                if (~isfield(info,'PCARatio')); info.PCARatio = 1; end; % keep all
                
                options.Fisherface = info.Fisherface; 
                options.PCARatio   = info.PCARatio;
            
            end
            
            % Perform LDA
            [ffaces, ~] = DengCai.LDA(XL, options, X'); % Raw major data matrix
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function W = r1_optimize(nndb) 
            %R1_OPTIMIZE: finds an optimal solution for LDA-R1 objective.
            
            % Dimension of a sample
            d = nndb.h * nndb.w;
            
            % Set working varibles
            X = nndb.features/255;
            n_per_class = double(unique(nndb.n_per_class));
            n = nndb.n;
            cls_n = nndb.cls_n;
            
            % Error handling
            if (numel(n_per_class) > 1)
                error(['DCC_OPT_ERR: Multiple samples per class (n_per_class) is not supported.']);
            end
            
            % [PERF] Summation Matrix for efficient calculation of class mean.
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
            U = (X * sum_MAT)/n_per_class;
            U0 = repmat(mean(U, 2), 1, cls_n);           

            % Initialize W
            W = eye(d, d);

            % Initialize alpha
            alpha = 0.5;
            
            % Info of previous iteration
            pW = W;
            prevErr = 0;

            % Initialize temporary cost variables
            inter_cost = zeros(d, d);
            intra_cost = zeros(d, d);
            
            % TT = reshape(U, 32, 32, 1, []);
            % imshow(TT(:, :, :, 1));

            while (true) 
                
                % Calculate inter cost
                % U does not have the same dimension as X
                for i = 1:cls_n
                    diff = (U(:, i) - U0(:, i));
                    C =  diff * diff';
                    inter_cost = inter_cost + (n_per_class * C / sqrt(trace(W' * C * W)));
                end
                                       
                % Calculate intra cost
                % U does not have the same dimension as X
                for i = 1:n
                    diff = (X(:, i) - U(:, idivide(uint16(i-1), n_per_class)+1));
                    C =  diff * diff';
                    intra_cost = intra_cost + (C / sqrt(trace(W' * C * W)));
                end
                                
                cost = (1-alpha) * inter_cost - alpha * intra_cost;
        
                [W, ~, ~] = svd(cost); 

                err = norm(W - pW);
                disp(['Error: ' num2str(err)]);
                if (err < 1e-5)
                    disp(['Break by error = 0']);
                    break;
                end

                if ((prevErr - err) < 1e-5)
                    disp(['Break by error diff = 0']);
                    break;
                end

                pW = W;
                prevErr = err;
            end
        end   
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [W, info] = dl2_optimize(nndb, info) 
            %DL2_OPTIMIZE: finds an optimal solution for LDA-R1 objective.
            
            % Set defaults for info fields, if the field does not exist  
            if (~isfield(info,'threhold_db')); info.threhold_db = 0.0001; end; 
            if (~isfield(info,'threhold_dw')); info.threhold_dw = 5; end; 
            
            % Dimension of a sample
            d = nndb.h * nndb.w;
            
            % Set working varibles
            X = nndb.features/255;
            n_per_class = double(unique(nndb.n_per_class));
            n = nndb.n;
            cls_n = nndb.cls_n;
            
            % Error handling
            if (numel(n_per_class) > 1)
                error(['DCC_OPT_ERR: Multiple samples per class (n_per_class) is not supported.']);
            end
            
            % [PERF] Summation Matrix for efficient calculation of class mean.
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
            U = (X * sum_MAT)/n_per_class;
            U0 = repmat(mean(U, 2), 1, cls_n);           
            
            
            %%% OPTIMIZE |Sb|/|Sw| %%%
            
            % PERF: Using svd need not calculate Sb
            % Sb = (U - U0)*(U - U0)';
            
            % P' * Sb * P = Db            
            [P, Db_h, ~] = svd(U - U0);
            
            % Db is already sorted after svd (decreasing order)
            % Preserve the largest values for Db
            db = diag(Db_h);
            info.db = db;
            ind = find(db > info.threhold_db); % TODO: take the percentages from the user
            if (~isempty(ind))
                en = ind(end); 
                Db_h = diag(db(1:en));
                P = P(:, 1:en);
            end
            
            % Unitize Sb
            Z = P/Db_h; % where Z' * Sb * Z = I
                       
            % same dimension as X
            U_X = reshape(repmat(U, n_per_class, 1), d, []);   
            
            % PERF: Using svd need not calculate Sw
            % Sw = (X - U_X)*(X - U_X)';
            
            % P' * (Z' * Sw * Z) * P = Dw                        
            [P, Dw_h, ~] = svd(Z' * (X - U_X));
            
            % Dw is already sorted after svd (decreasing order)
            % Preserve the smallest values for Dw
            dw = diag(Dw_h);
            dw = sort(dw, 'descend');
            info.dw = dw;
            ind = find(dw < info.threhold_dw);  % TODO: take the percentages from the user
            if (~isempty(ind))
                en = ind(end); 
                Dw_h = diag(dw(1:en));
                P = P(:, 1:en);
            end
            
            % 'A' diagionalizes the Sw, Sb
            A = P' * Z';
            
            % assert: A * Sw * A' = Dw;
            % assert: A * Sb * A' = I
            
            % Speheres the data (unit variance in the projected space)
            W = (Dw_h\A)'; % inv(Db_h) * A
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end    
end

