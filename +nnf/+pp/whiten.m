function [X, P, M] = whiten(X, fudgefactor)
    % X: row major matrix
    % ZCA whitening
       
    C = cov(X);
    M = mean(X);
    
%     Xc = bsxfun(@minus, X, M)
%     [V,D] = eig(Xc'*Xc);

    [V,D] = eigs(C); % TODO: check the difference between eig and eigs (eigs: more stable)
    
    % In PCA space
    P = V * diag(sqrt(1./(diag(D) + fudgefactor)));
    
    % ZCA whitening, (rotating data after whiten operation)
    % P = V * diag(sqrt(1./(diag(D) + fudgefactor))) * V'; % rotating it back
    
    
    X = bsxfun(@minus, X, M) * P;
end