classdef NNPatch
    % NNPatch describes the patch information.
    %
    % i.e
    % Patch of 33x33 at position (1, 1)
    % nnpatch = NNPatch(33, 33, [1 1])
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
        
    properties (SetAccess = public)
        w;      % Width        
        h;      % Height                 
        offset; % Position of the patch     
    end
    
    properties (SetAccess = private)
        user_data % (internal use) Hold nndbtr, nndbval, nndbte patch databases in DL framework
    end
    
    methods (Access = public)
        function self = NNPatch(width, height, offset)
            
            % Matlab support preinitializing the object arrays
            % Ref: https://au.mathworks.com/help/matlab/matlab_oop/initialize-object-arrays.html
            if (nargin <= 0); return; end;
            
            % Initialize the variables
            self.w = width;
            self.h = height;
            self.offset = offset;
            self.user_data = [];
        end
    end
end