classdef PCA < nnf.alg.MCC
    % PCA: Principle Component Analysis algorithm and varients.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - PCA.l2 
    %   - PCA.l1
    %   - PCA.mcc    
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties 
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [W, m] = l2(nndb, info) 
            % L2: leanrs the PCA subspace with l2-norm. 
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
            %     inf.ReducedDim = 0;     % No of dimension (0 = keep all)  
            %     inf.visualize  = false; % Visualize eigen faces
            % 
            % Returns
            % -------
            % W : 2D matrix -double
            %     Projecttion matrix.
         	%
            % m : vector -double
            %     Mean vector.
            %
            % Examples
            % --------
            % import nnf.alg.PCA;
            % W = PCA.l2(nndb_tr)
            %
                        
            % Imports 
            import nnf.alg.PCA;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end
                                
            % Fetch eigen faces
            [W,~,m] = PCA.eig_face_core(nndb.features, info);
            
            % Visualize eigen faces if required
            if (isfield(info,'visualize') && info.visualize)                
                imdb = uint8(reshape(W*255 + min(min(W) + 10), nndb.h, nndb.w, nndb.ch, []));
                figure, immap(imdb, 3, 5);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      	function W = l1(X, no_dim) 
            % TODO: SLINDA
            
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function W = mcc(nndb, info) 
            % MCC: leanrs the MCC-PCA subspace with l2-norm. 
            %
            % Parameters
            % ----------
            % nndb : nnf.db.NNdb
            %     Data object that contains the database.
            % 
            % info : struct, optional
            %     Provide additional information to perform MCC-PCA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     info.ReducedDim = 0;     % No of dimension (TODO: implement)
            %     info.visualize  = false; % Visualize mcc faces
            % 
            %
            % Returns
            % -------
            % W : 2D matrix -double
            %     Projecttion matrix.
         	%
            %
            % Examples
            % --------
            % import nnf.alg.PCA;
            % W = PCA.mcc(nndb_tr)
            %
            
            % Imports 
            import nnf.alg.PCA;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end

            W = PCA.mcc_optimize(nndb);
            
            % Visualize mcc faces if required
            if (isfield(info,'visualize') && info.visualize)                
                imdb = uint8(reshape(W*255 + min(min(W) + 10), nndb.h, nndb.w, nndb.ch, []));
                figure, immap(imdb, 3, 5);
            end

        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndb_te_r = l2_reconst(W, m, nndb_te, info) 
            % L2_RECONST: Reconstruct occluded faces with PCA (l2-norm). 
            %   Preconditions: Occlusion filter `info.oc_filter` must be provided.
            %
            % Parameters
            % ----------
            % W : 2D matrix -double
            %     Projecttion matrix.
            % 
            % m : vector -double
            %     Mean vector of the training database.
            % 
            % nndb_te : nnf.db.NNdb
            %     Data object that contains the te database.
            %
            % info : struct
            %     Provide additional information to perform PCA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.ReducedDim = 0;     % No of dimension (0 = keep all)  
            %     inf.visualize  = false; % Visualize eigen faces
            % 
            %     % Occlusion filter properties
            %     inf.oc_filter.percentage = 0; % Percentage
            %     inf.oc_filter.type = 0;       % type ('t', 'b', 'l', 'r')
            %
            %
            % Returns
            % -------
            % W : 2D matrix -double
            %     Projecttion matrix.
         	%
            %
            % Examples
            % --------
            % import nnf.alg.PCA;
            % W = PCA.l2(nndb_tr)
            %
                        
            % Imports            
            import nnf.db.NNdb;
            import nnf.db.DbSlice;
            import nnf.alg.PCA;
            import nnf.utl.immap;
            
            % Set defaults for arguments
            if (nargin < 3), info = []; end
            assert(isfield(info,'oc_filter'));
            assert(isfield(info.oc_filter,'percentage'));
            assert(isfield(info.oc_filter,'offset'));
                                            
            % Fetch eigen faces
            % [W,~,m] = PCA.eig_face_core(nndb_tr.features, info);
                        
            % Reconstruction
            occl_rate = info.oc_filter.percentage;
            if (isfield(info.oc_filter,'type')) 
                occl_type = info.oc_filter.type;
            else
                occl_type = ''; % empty defaults to 'b'
            end
            occl_offset = info.oc_filter.offset;
                        
            h = nndb_te.h;
            w = nndb_te.w;
            ch = nndb_te.ch;            
            
            filter = DbSlice.get_occlusion_patch(h, w, class(nndb_te.db), occl_type, occl_rate, occl_offset);                                                                                
            filter = repmat(filter, 1, 1, ch);
            filter = double(diag(filter(:)));
            S = pinv(transpose((transpose(filter*W)*(filter*W))))*transpose(filter*W)* ...
                bsxfun(@minus, filter*nndb_te.features, filter*m);
            
            db_te_r = bsxfun(@plus, W * S, m);
            db_te_r = reshape(uint8(db_te_r), h, w, ch, []);
            nndb_te_r = NNdb('te_reconst', db_te_r, nndb_te.n_per_class, nndb_te.build_cls_lbl, nndb_te.cls_lbl, nndb_te.db_format);
            
            % Visualize eigen faces if required
            if (isfield(info,'visualize') && info.visualize)
                imdb = uint8(reshape(W*255 + min(min(W) + 10), nndb.h, nndb.w, nndb.ch, []));
                figure, immap(imdb, 3, 5);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    end
    
    methods (Access = private, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [efaces, evalues, sample_mean] = eig_face_core(X, info) 
            %EIG_FACE_CORE: performs eigen face learning.
            
            % Imports
            import nnf.libs.DengCai;
            
            % Set defaults for info fields, if the field does not exist   
            if (~isfield(info,'ReducedDim')); info.ReducedDim = 0; end;  % Keep all  
            
            % Perform PCA
            options.ReducedDim = info.ReducedDim;   
            [efaces, evalues, sample_mean] = DengCai.PCA(X', options); % Raw major data matrix
            sample_mean = sample_mean';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function W = mcc_optimize(nndb) 
            %MCC_OPTIMIZE: finds an optimal solution for MCC-PCA objective.
            
            % Imports 
            import nnf.alg.MCC;
            import nnf.utl.*;
            
            % Small value
            delta = 0.0001;
                        
            % Random orthogonal projection matrix
            W = orth(rand(nndb.h*nndb.w, 70));% n-10)); %size(X)));
           
            % Calculate mean, mean centred data
            X = nndb.features/255;
            M = mean(X);
            Xc = bsxfun(@minus, X, M);
            
            % Previous cumulated error
            prev_cerr = 0;
            
            % No. of samples
            n = size(X, 2);
            
            while (1)
                % Error vector
                pca_err = diag(Xc'*Xc - Xc'*W*(W')*Xc);

                % Dynamic sigma
                sigma_2 = MCC.calc_sigma_s(col_norm_sqrd(Xc'*Xc - Xc'*W*(W')*Xc), n);

                % Gauss error
                g_err = MCC.gauss(pca_err/sigma_2);
                p = -g_err;    

                % Calculate cumilated error (Max value = n: no of samples)
                cerr = sum(-p);
                
                % Display min error
                disp(num2str(n - cerr));
                
                % Break if convergence is met
                if ((n - cerr < delta) || (abs(prev_cerr-cerr) < 1e-6))
                   break;
                end    
                prev_cerr = cerr;

                % Calculate new mean (weighted by p)
                M = X * p/sum(p);
                
                % Calculate new mean centred data
                Xc = bsxfun(@minus, X, M);

                % P matrix
                P = diag(-p);

                % Eigen value decomposition of (Xc * P * Xc') for large dimension matrices
                Cov = sqrt(P)' * (Xc') * Xc * sqrt(P);
                [U, S] = svd(Cov);
                inv_s = pinv(S);
                W = Xc * sqrt(P)' * U * inv_s(:, 1:30);        
                %[UT2, ST2] = svd( Xc * P * Xc' );

                % IMPORTANT: Stable eigen decomp. functions: eigs, svds, svd 
                % (DO NOT USE eig -  not stable)
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end