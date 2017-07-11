function [PX, W, r_info] = whiten(X, fudgefactor, keepdims, squashtozero, threshold)
    % X: column major matrix    
     
    % Set defaults for arguments 
    if (nargin < 3);  keepdims = false; end
    if (nargin < 4);  squashtozero = false; end
    if (nargin < 5);  threshold = 1e-10; end  
   
    % Calculate Covariance
    m = mean(X, 2);    
    Xc = bsxfun(@minus, X, m);

    % Eigen decomposition
    % [V, D] = eig(Xc*Xc');
    
    % Stable eigen value decomposition
    [V, D0] = svd(Xc);
    %D = D0 .^ 2;    
    D = D0;
    
    d = diag(D);        
    en = numel(d);
    
	if squashtozero == true
		d(d < threshold) = 0;
	end
	
    if keepdims == false
        d(d < threshold) = 0;
        en = find(~d);
        if (~isempty(en) && en(1)-1 ~= 0)
            en = en(1)-1;
        end
    end

    %DD = diag(sqrt(1./(d + fudgefactor)));
    DD = diag(1./(d + fudgefactor));
    DD = DD(:, 1:en);
    
    % For Under complete dictionay
    DD = [DD; zeros(size(D, 1) - size(DD, 1), size(DD, 2))];
       
    W = V * DD;
    PX = W' * Xc;
    
    % Save the info to reverse the whitening process
    r_info.m = m;
    r_info.V = V;
    r_info.d = d;
    r_info.fudgefactor = fudgefactor;
    r_info.en = en;
    
    % ZCA whitening
    % Rotating back to the original space
    %W = V * diag(sqrt(1./(diag(D) + fudgefactor))) * V';   
end