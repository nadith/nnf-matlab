classdef TanhHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % TanhGPUStrategy   Execution strategy for running Tanh on the host
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X)
            Z = tanhForward(X);
            memory = [];
        end
        
        function [dX,dW] = backward(~, Z, dZ, X)
            dX = tanhBackward(Z, dZ, X);
            dW = [];
        end
    end
end