classdef Util
    %UTIL: Utility class to provide common utility functions for algorithms.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - test      (evaluates classification accurary in the given subspace) 
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
    end
    
    methods(Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function [accuracy, dist] = PLS_test(W1, nndb_g1, W2, nndb_g2, nndb_p, info)
            % TEST: evaluates classification accurary in the given subspace. 
            %
            % Parameters
            % ----------
            % nndb_g : nnf.db.NNdb
            %     Gallery database object.
            % 
            % nndb_p : nnf.db.NNdb
            %     Probe database object.
            % 
            % info : struct, optional
            %     Provide additional information to perform test. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.dist     = false;     % Calculate distance matrix. TODO: implement
            %     inf.dcc_norm = false;     % Distance metric norm to be used. (false = Use L2 norm)
            %     inf.dcc_sigma= eye(n, n); % Sigma Matrix that describes kernel bandwidth for each class. 
            % 
            % 
            % Returns
            % -------
            % accuracy : double
            %     Classification accuracy.
         	%
            % dist : 2D matrix -double
            %     Distance matrix. (Tr.Samples x Te.Samples)
            %
            %
            % Examples
            % --------
            % import nnf.alg.PCA;
            % import nnf.alg.Util;
            % W = PCA.l2(nndb_tr)
            % accurary = Util.test(W, nndb_tr, nndb_te)
            %
        
            % Imports
            import nnf.utl.col_norm_sqrd;
            
            % Set defaults for arguments
            if (nargin < 6), info = []; end
            if (~isfield(info,'perf')); info.perf = []; end; 
            if (~isfield(info.perf,'use_mem')); info.perf.use_mem = false; end; 
            if (~isfield(info,'g_norm')); info.g_norm = false; end; 
         
            % TODO: Implement
            % % Initialize distance matrix to capture the distance
            % if (isfield(info,'dist'))
            %     assert(numel(unique(nndb_g.n_per_class)) == 1); % All classes has same n_per_class
            %     dist = zeros(nndb_g.n_per_class(1), nndb_p.n);
            % end           
            
            % Project the gallery/probe images
            gp_X =  [W1' * nndb_g1.features  W2' * nndb_g2.features]; % gallery images
            p_XT =  W2' * nndb_p.features; % probe images
                   
            % Extended gallery class labels and 'n'
            g_cls_lbl = [nndb_g1.cls_lbl nndb_g2.cls_lbl];
            g_n = nndb_g1.n + nndb_g2.n;
            
            % To store the verification result
            v = uint8(zeros(1, nndb_p.n));

            % Fetch feature size
            [fsize, ~] = size(p_XT);
            
            for te_idx = 1:nndb_p.n

                gp_diff = zeros(1, g_n);
                
                % Small memory foot print, high CPU overhead
                for tr_idx = 1:g_n 
                    % Calculate the norm according to the preference
                    if (info.g_norm)
                        %TODO: Complete
                        gp_diff(tr_idx) = 1 - DCC_Gauss(ColumnNormSqrd(gp_X(:, tr_idx) - p_XT(:, te_idx))*sigma(i));
                    else
                        gp_diff(tr_idx) = col_norm_sqrd(gp_X(:, tr_idx) - p_XT(:, te_idx)); % Eucludien distance
                    end
                end
                
                % Fetch the minimum measure (distance)
                [mn, idx] = min(gp_diff);

                % Save the distance if required
                if (isfield(info,'dist'))
                    dist(:, te_idx) = gp_diff';
                end
                
                % Set the verification result
                v(te_idx) = ((g_cls_lbl(idx) == nndb_p.cls_lbl(te_idx)));
            end

            % Calculate accuracy
            accuracy = (sum(v) / nndb_p.n) * 100;
        end
        
        function [accuracy, dist] = test(W, nndb_g, nndb_p, info) 
            % TEST: evaluates classification accurary in the given subspace. 
            %
            % Parameters
            % ----------
            % nndb_g : nnf.db.NNdb
            %     Gallery database object.
            % 
            % nndb_p : nnf.db.NNdb
            %     Probe database object.
            % 
            % info : struct, optional
            %     Provide additional information to perform test. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     inf.dist     = false;     % Calculate distance matrix. TODO: implement
            %     inf.dcc_norm = false;     % Distance metric norm to be used. (false = Use L2 norm)
            %     inf.dcc_sigma= eye(n, n); % Sigma Matrix that describes kernel bandwidth for each class. 
            % 
            % 
            % Returns
            % -------
            % accuracy : double
            %     Classification accuracy.
         	%
            % dist : 2D matrix -double
            %     Distance matrix. (Tr.Samples x Te.Samples)
            %
            %
            % Examples
            % --------
            % import nnf.alg.PCA;
            % import nnf.alg.Util;
            % W = PCA.l2(nndb_tr)
            % accurary = Util.test(W, nndb_tr, nndb_te)
            %
        
            % Imports
            import nnf.utl.col_norm_sqrd;
            
            % Set defaults for arguments
            if (nargin < 4), info = []; end
            if (~isfield(info,'perf')); info.perf = []; end; 
            if (~isfield(info.perf,'use_mem')); info.perf.use_mem = false; end; 
            if (~isfield(info,'g_norm')); info.g_norm = false; end; 
            
            % Initialize distance matrix to capture the distance
            if (isfield(info,'dist'))
                assert(numel(unique(nndb_g.n_per_class)) == 1); % All classes has same n_per_class
                dist = zeros(nndb_p.n, nndb_g.n);
            end           
            
            % Project the gallery images
            if ((isfield(info, 'use_kda') && info.use_kda) || ...
                    (isfield(info, 'use_kpca') && info.use_kpca))
                KD_tr = constructKernel(nndb_g.features', [],  info.koptions);
                gp_X = W' * KD_tr';                
            else
                gp_X =  W' * nndb_g.features; % - m (mean used in training); % optional center the gallery images
            end
            
            % Project the test images 
            if ((isfield(info, 'use_kda') && info.use_kda) || ...
                    (isfield(info, 'use_kpca') && info.use_kpca))
                KD_te = constructKernel(nndb_p.features', nndb_g.features',  info.koptions);
                p_XT = W' * KD_te';                
            else
                p_XT =  W' * nndb_p.features; % - m (mean used in training); % optional center the probe images
            end
            
            if (info.perf.use_mem)
                % PERF: Iteration of each gallery image to calculate distance to test image is
                % eliminated.
                % Reduce CPU overhead, but large memory foot print
                gp_XT = repmat(p_XT, nndb_g.n, 1);
            end
                      
            % To store the verification result
            v = uint8(zeros(1, nndb_p.n));

            % Fetch feature size
            [fsize, ~] = size(p_XT);
            
            for te_idx = 1:nndb_p.n

                gp_diff = zeros(1, nndb_g.n);
                
                if (info.perf.use_mem)
                    % Large memory foot print, less CPU overhead
                    
                    % Same dimension as gp_X, compatiability for matrix operations
                    gp_XT1 = reshape(gp_XT(:, te_idx), fsize, []);

                    if (info.g_norm)
                        %TODO: Complete
                        gp_diff = 1 - DCC_Gauss(ColumnNormSqrd(gp_X - gp_XT1)*sigma(i));
                    else
                        gp_diff = col_norm_sqrd(gp_X - gp_XT1); % Eucludien distance
                    end
                        
                else
                    
                    % Small memory foot print, high CPU overhead
                    for tr_idx = 1:nndb_g.n 
                        % Calculate the norm according to the preference
                        if (info.g_norm)
                            %TODO: Complete
                            gp_diff(tr_idx) = 1 - DCC_Gauss(ColumnNormSqrd(gp_X(:, tr_idx) - p_XT(:, te_idx))*sigma(i));
                        else
                            gp_diff(tr_idx) = col_norm_sqrd(gp_X(:, tr_idx) - p_XT(:, te_idx)); % Eucludien distance
                        end
                    end
                    
                end
                
                % Fetch the minimum measure (distance)
                [mn, idx] = min(gp_diff);

                % Save the distance if required
                if (isfield(info,'dist'))
                    dist(te_idx, :) = gp_diff;
                end                

                % Set the verification result
                v(te_idx) = ((nndb_g.cls_lbl(idx) == nndb_p.cls_lbl(te_idx)));

            end

            % Calculate accuracy
            accuracy = (sum(v) / nndb_p.n) * 100;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

