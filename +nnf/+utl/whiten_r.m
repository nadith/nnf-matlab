function [X] = whiten_r(PX, r_info)   

    % Information needed for whitening reverse 
    m = r_info.m;
    V = r_info.V;
    d = r_info.d;
    fudgefactor = r_info.fudgefactor;
    en = r_info.en;

    % Reverse %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = diag(d);
    DD = diag(d + fudgefactor);
    DD = DD(:, 1:en);

    % For Under complete dictionay
    DD = [DD; zeros(size(D, 1) - size(DD, 1), size(DD, 2))];
    
    W = V * DD;
    Xc = W * PX;
    X = bsxfun(@plus, Xc, m);
end