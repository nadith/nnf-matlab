function [ Y ] = illumination_norm( X )
    % NORM_ILLUMINATION: Normalize illumination in image X.
    % Note that following normalization techniques in `inface` library are only compatible with 
    % grayscale images.

    dtype = class(X);
    X = normalize8(X);
    % X = normalize8(imresize(X,[128,128],'bilinear'));  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Gradientfaces      
    % Y = gradientfaces(X, 0.3); %reflectance
    % Y = rank_normalization(Y);
        
    % IS
    % Y = isotropic_smoothing(X); %reflectance
    % Y = rank_normalization(Y);
    
    % WA
    Y = wavelet_normalization(X); %reflectance
    Y = rank_normalization(Y);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Y = normalize8(Y);
    Y = cast(Y, dtype);
end

