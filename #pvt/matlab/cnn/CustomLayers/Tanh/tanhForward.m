function Z = tanhForward(X)
% tanhForward   Rectified Linear Unit (Tanh) activation on the host
%   Z = tanhForward(X) takes the input X and applies the Tanh function to
%   return Z.
%
%   Input:
%   X - Input channels for a set of images. A (H)x(W)x(C)x(N) array.
%
%   Output:
%   Z - Output channels for a set of images. A (H)x(W)x(C)x(N) array.

%   Copyright 2016-2017 The MathWorks, Inc.

%Z = max(0,X);
Z = 1.7159*tanh(2/3.*X);
end