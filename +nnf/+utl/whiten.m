function [PX, W, m] = whiten(X, fudgefactor)
    % X: column major matrix
    
    % Calculate Covariance
    m = mean(X, 2);    
    Xc = bsxfun(@minus, X, m);

    % Eigen decomposition
    % [V, D] = eig(Xc*Xc');
    
    % Stable eigen value decomposition
    [V, D0] = svd(Xc);
    D = D0 .^ 2;    
    
    W = V * diag(sqrt(1./(diag(D) + fudgefactor)));
    PX = W' * Xc;
    
    % ZCA whitening
    % Rotating back to the original space
    %W = V * diag(sqrt(1./(diag(D) + fudgefactor))) * V';   
end