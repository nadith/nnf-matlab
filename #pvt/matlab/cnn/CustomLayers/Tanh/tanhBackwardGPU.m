function dLossdX = tanhBackwardGPU(Z, dLossdZ, X) %#ok<INUSL>
% tanhBackward   Perform backpropagation for a ReLU layer
%
% Inputs:
% Z - The output from the ReLU layer. Unused, here to match cuDNN API.
% dLossdZ - The derivative of the loss function with respect to the output
% of the ReLU layer. A (H)x(W)x(C)x(N) array.
% X - The input to the ReLU layer. A (H)x(W)x(C)x(N) array.
%
% Output:
% dLossdX - The derivative of the loss function with respect to the input
% of the ReLU layer. A (H)x(W)x(C)x(N) array.

%   Copyright 2015-2016 The MathWorks, Inc.

% Ensure calculation on GPU even for host-side inputs
% X = gpuArray(X); % No-op if already on GPU
dLossdZ = gpuArray(dLossdZ); % No-op if already on GPU
Z = gpuArray(Z); 

% dLossdX = dLossdZ .* (X > 0);


d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * Z.^2);
dLossdX = dLossdZ .* d_act;
end

% function dX = leakyReluBackward(~, dZ, X, scale)
% % Back-propagation using Leaky Rectified Linear on the GPU
% 
% %   Copyright 2016 The MathWorks, Inc.
% 
% % Ensure calculation on GPU even for host-side inputs
% X = gpuArray(X); % No-op if already on GPU
% dZ = gpuArray(dZ); % No-op if already on GPU
% 
% % Now scale down the negative inputs
% negVals = (X < 0);
% dX = dZ - negVals .* (1-scale) .* dZ; % Avoid indexing for speed
% 
% end
