classdef DRToolBox
    %DRTOOLBOX: Lists the interface for the DRToolBox library.
    %   Refer method specific help for more details.        
    %
    %   Currently Support:
    %   ------------------
    %   - DRToolBox.lle
    %   - DRToolBox.tsne
    %   - DRToolBox.tsne_p    
    
    properties
    end
    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [mappedX, mapping] = lle(X, no_dims, k, eig_impl) 
            %LLE Runs the locally linear embedding algorithm
            %
            %   mappedX = lle(X, no_dims, k, eig_impl)
            %
            % Runs the local linear embedding algorithm on dataset X to reduces its
            % dimensionality to no_dims. In the LLE algorithm, the number of neighbors
            % can be specified by k. 
            % The function returns the embedded coordinates in mappedX.
            %
            %

            % This file is part of the Matlab Toolbox for Dimensionality Reduction.
            % The toolbox can be obtained from http://homepage.tudelft.nl/19j49
            % You are free to use, change, or redistribute this code in any way you
            % want for non-commercial purposes. However, it is appreciated if you 
            % maintain the name of the original author.
            %
            % (C) Laurens van der Maaten, Delft University of Technology


                if ~exist('no_dims', 'var')
                    no_dims = 2;
                end
                if ~exist('k', 'var')
                    k = 12;
                end
                if ~exist('eig_impl', 'var')
                    eig_impl = 'Matlab';
                end

                % Get dimensionality and number of dimensions
                [n, d] = size(X);

                % Compute pairwise distances and find nearest neighbors (vectorized implementation)
                disp('Finding nearest neighbors...');    
                [distance, neighborhood] = find_nn(X, k);

                % Identify largest connected component of the neighborhood graph
                blocks = components(distance)';
                count = zeros(1, max(blocks));
                for i=1:max(blocks)
                    count(i) = length(find(blocks == i));
                end
                [count, block_no] = max(count);
                conn_comp = find(blocks == block_no); 

                % Update the neighborhood relations
                tmp = 1:n;
                tmp = tmp(conn_comp);
                new_ind = zeros(n, 1);
                for i=1:n
                    ii = find(tmp == i);
                    if ~isempty(ii), new_ind(i) = ii; end
                end 
                neighborhood = neighborhood(conn_comp,:)';
                for i=1:n
                    neighborhood(neighborhood == i) = new_ind(i);
                end
                n = numel(conn_comp);
                X = X(conn_comp,:)';    
                max_k = size(neighborhood, 1);

                % Find reconstruction weights for all points by solving the MSE problem 
                % of reconstructing a point from each neighbours. A used constraint is 
                % that the sum of the reconstruction weights for a point should be 1.
                disp('Compute reconstruction weights...');
                if k > d 
                    tol = 1e-5;
                else
                    tol = 0;
                end

                % Construct reconstruction weight matrix
                W = zeros(max_k, n);
                for i=1:n
                    nbhd = neighborhood(:,i);
                    nbhd = nbhd(nbhd ~= 0);
                    kt = numel(nbhd);
                    z = bsxfun(@minus, X(:,nbhd), X(:,i));                  % Shift point to origin
                    C = z' * z;												% Compute local covariance
                    C = C + eye(kt, kt) * tol * trace(C);					% Regularization of covariance (if K > D)
                    wi = C \ ones(kt, 1);                                   % Solve linear system
                    wi = wi / sum(wi);                                      % Make sure that sum is 1
                    W(:,i) = [wi; nan(max_k - kt, 1)];
                end

                % Now that we have the reconstruction weights matrix, we define the 
                % sparse cost matrix M = (I-W)'*(I-W).
                M = sparse(1:n, 1:n, ones(1, n), n, n, 4 * max_k * n);
                for i=1:n
                   w = W(:,i);
                   j = neighborhood(:,i);
                   indices = find(j ~= 0 & ~isnan(w));
                   j = j(indices);
                   w = w(indices);
                   M(i, j) = M(i, j) - w';
                   M(j, i) = M(j, i) - w;
                   M(j, j) = M(j, j) + w * w';
                end

                % For sparse datasets, we might end up with NaNs or Infs in M. We just set them to zero for now...
                M(isnan(M)) = 0;
                M(isinf(M)) = 0;

                % The embedding is computed from the bottom eigenvectors of this cost matrix
                disp('Compute embedding (solve eigenproblem)...');
                tol = 0;
                if strcmp(eig_impl, 'JDQR')
                    options.Disp = 0;
                    options.LSolver = 'bicgstab';
                    [mappedX, eigenvals] = jdqr(M + eps * eye(n), no_dims + 1, tol, options);
                else
                    options.disp = 0;
                    options.isreal = 1;
                    options.issym = 1;
                    [mappedX, eigenvals] = eigs(M + eps * eye(n), no_dims + 1, tol, options);          % only need bottom (no_dims + 1) eigenvectors
                end
                [eigenvals, ind] = sort(diag(eigenvals), 'ascend');
                if size(mappedX, 2) < no_dims + 1
                    no_dims = size(mappedX, 2) - 1;
                    warning(['Target dimensionality reduced to ' num2str(no_dims) '...']);
                end
                eigenvals = eigenvals(2:no_dims + 1);
                mappedX = mappedX(:,ind(2:no_dims + 1));                                % throw away zero eigenvector/value

                % Save information on the mapping
                mapping.k = k;
                mapping.X = X';
                mapping.vec = mappedX;
                mapping.val = eigenvals;
                mapping.conn_comp = conn_comp;
                mapping.nbhd = distance;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function ydata = tsne(X, labels, no_dims, initial_dims, perplexity, max_iter) 
            %TSNE Performs symmetric t-SNE on dataset X
            %
            %   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
            %   mappedX = tsne(X, labels, initial_solution, perplexity)
            %
            % The function performs symmetric t-SNE on the NxD dataset X to reduce its 
            % dimensionality to no_dims dimensions (default = 2). The data is 
            % preprocessed using PCA, reducing the dimensionality to initial_dims 
            % dimensions (default = 30). Alternatively, an initial solution obtained 
            % from an other dimensionality reduction technique may be specified in 
            % initial_solution. The perplexity of the Gaussian kernel that is employed 
            % can be specified through perplexity (default = 30). The labels of the
            % data are not used by t-SNE itself, however, they are used to color
            % intermediate plots. Please provide an empty labels matrix [] if you
            % don't want to plot results during the optimization.
            % The low-dimensional data representation is returned in mappedX.
            %
            %

            % This file is part of the Matlab Toolbox for Dimensionality Reduction.
            % The toolbox can be obtained from http://homepage.tudelft.nl/19j49
            % You are free to use, change, or redistribute this code in any way you
            % want for non-commercial purposes. However, it is appreciated if you 
            % maintain the name of the original author.
            %
            % (C) Laurens van der Maaten, Delft University of Technology

            % Imports
            import nnf.libs.DRToolBox;
            
            if ~exist('labels', 'var')
                labels = [];
            end
            if ~exist('no_dims', 'var') || isempty(no_dims)
                no_dims = 2;
            end
            if ~exist('initial_dims', 'var') || isempty(initial_dims)
                initial_dims = min(50, size(X, 2));
            end
            if ~exist('perplexity', 'var') || isempty(perplexity)
                perplexity = 30;
            end

            % First check whether we already have an initial solution
            if numel(no_dims) > 1
                initial_solution = true;
                ydata = no_dims;
                no_dims = size(ydata, 2);
                perplexity = initial_dims;
            else
                initial_solution = false;
            end

            % Normalize input data
            X = X - min(X(:));
            X = X / max(X(:));
            X = bsxfun(@minus, X, mean(X, 1));

            % Perform preprocessing using PCA
            if ~initial_solution
                disp('Preprocessing data using PCA...');
                if size(X, 2) < size(X, 1)
                    C = X' * X;
                else
                    C = (1 / size(X, 1)) * (X * X');
                end
                [M, lambda] = eig(C);
                [lambda, ind] = sort(diag(lambda), 'descend');
                M = M(:,ind(1:initial_dims));
                lambda = lambda(1:initial_dims);
                if ~(size(X, 2) < size(X, 1))
                    M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
                end
                X = X * M;
                clear M lambda ind
            end

            % Compute pairwise distance matrix
            sum_X = sum(X .^ 2, 2);
            D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

            % Compute joint probabilities
            P = d2p(D, perplexity, 1e-5);                                           % compute affinities using fixed perplexity
            clear D

            % Run t-SNE
            if initial_solution
                ydata = DRToolBox.tsne_p(P, labels, ydata, max_iter);
            else
                ydata = DRToolBox.tsne_p(P, labels, no_dims, max_iter);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function ydata = tsne_p(P, labels, no_dims, max_iter) 
            %TSNE_P Performs symmetric t-SNE on affinity matrix P
            %
            %   mappedX = tsne_p(P, labels, no_dims)
            %
            % The function performs symmetric t-SNE on pairwise similarity matrix P 
            % to create a low-dimensional map of no_dims dimensions (default = 2).
            % The matrix P is assumed to be symmetric, sum up to 1, and have zeros
            % on the diagonal.
            % The labels of the data are not used by t-SNE itself, however, they 
            % are used to color intermediate plots. Please provide an empty labels
            % matrix [] if you don't want to plot results during the optimization.
            % The low-dimensional data representation is returned in mappedX.
            %
            %

            % This file is part of the Matlab Toolbox for Dimensionality Reduction.
            % The toolbox can be obtained from http://homepage.tudelft.nl/19j49
            % You are free to use, change, or redistribute this code in any way you
            % want for non-commercial purposes. However, it is appreciated if you 
            % maintain the name of the original author.
            %
            % (C) Laurens van der Maaten, Delft University of Technology

            if ~exist('labels', 'var')
                labels = [];
            end
            if ~exist('no_dims', 'var') || isempty(no_dims)
                no_dims = 2;
            end

            % First check whether we already have an initial solution
            if numel(no_dims) > 1
                initial_solution = true;
                ydata = no_dims;
                no_dims = size(ydata, 2);
            else
                initial_solution = false;
            end

            % Initialize some variables
            n = size(P, 1);                                     % number of instances
            momentum = 0.5;                                     % initial momentum
            final_momentum = 0.8;                               % value to which momentum is changed
            mom_switch_iter = 250;                              % iteration at which momentum is changed
            stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
            %max_iter = 3000;                                    % maximum number of iterations
            epsilon = 500;                                      % initial learning rate
            min_gain = .01;                                     % minimum gain for delta-bar-delta

            % Make sure P-vals are set properly
            P(1:n + 1:end) = 0;                                 % set diagonal to zero
            P = 0.5 * (P + P');                                 % symmetrize P-values
            P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
            const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
            if ~initial_solution
                P = P * 4;                                      % lie about the P-vals to find better local minima
            end

            % Initialize the solution
            if ~initial_solution
                ydata = .0001 * randn(n, no_dims);
            end
            y_incs  = zeros(size(ydata));
            gains = ones(size(ydata));

            % Run the iterations
            for iter=1:max_iter

                % Compute joint probability that point i and j are neighbors
                sum_ydata = sum(ydata .^ 2, 2);
                num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
                num(1:n+1:end) = 0;                                                 % set diagonal to zero
                Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities

                % Compute the gradients (faster implementation)
                L = (P - Q) .* num;
                y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;

                % Update the solution
                gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
                      + (gains * .8) .* (sign(y_grads) == sign(y_incs));
                gains(gains < min_gain) = min_gain;
                y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
                ydata = ydata + y_incs;
                ydata = bsxfun(@minus, ydata, mean(ydata, 1));

                % Update the momentum if necessary
                if iter == mom_switch_iter
                    momentum = final_momentum;
                end
                if iter == stop_lying_iter && ~initial_solution
                    P = P ./ 4;
                end

                % Print out progress
                if ~rem(iter, 10)
                    cost = const - sum(P(:) .* log(Q(:)));
                    disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
                end

                % Display scatter plot (maximally first three dimensions)
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, ydata, 9, labels, 'filled');
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled');
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled');
                    end
                    axis equal tight
        %             axis off
                    drawnow
                end
            end
        end    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end



