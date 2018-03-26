classdef TanhGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % TanhGPUStrategy   Execution strategy for running Tanh on the GPU
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = tanhForwardGPU(X);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X)
            dX = tanhBackwardGPU(Z, dZ, X);
            dW = [];
        end
    end
end