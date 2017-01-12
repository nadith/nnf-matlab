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
    
    d = diag(D);
    d(d<1e-10) = 0;
       
    en = find(~d);
    if (~isempty(en) && en(1)-1 ~= 0)
        en = en(1)-1;
    else
        en = numel(d);
    end
    
    DD = diag(sqrt(1./(d + fudgefactor)));
    DD = DD(:, 1:en);
    
    W = V * DD;
    PX = W' * Xc;
    
    % ZCA whitening
    % Rotating back to the original space
    %W = V * diag(sqrt(1./(diag(D) + fudgefactor))) * V';   
end