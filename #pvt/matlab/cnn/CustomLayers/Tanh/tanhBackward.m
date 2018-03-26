function dLossdX = tanhBackward(Z, dLossdZ, X) %#ok<INUSL>
% tanhBackward   Perform backpropagation for a Tanh layer
%
% Inputs:
% Z - The output from the Tanh layer. Unused, here to match cuDNN API.
% dLossdZ - The derivative of the loss function with respect to the output
% of the Tanh layer. A (H)x(W)x(C)x(N) array.
% X - The input to the Tanh layer. A (H)x(W)x(C)x(N) array.
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the Tanh layer. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2016 The MathWorks, Inc.

% dLossdX = dLossdZ .* (X > 0);

d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * Z.^2);
dLossdX = dLossdZ .* d_act;
end
