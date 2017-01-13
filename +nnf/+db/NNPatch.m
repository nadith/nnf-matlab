classdef NNPatch
    %NNPatch Enumeration describes the different noise types.    
    
    properties (SetAccess = public)
        w;            
        h;                 
        offset; 
    end
    
    properties (SetAccess = private)
        nndbtr; 
        nndbval; 
        nndbte;
    end
    
    methods (Access = public)
        function self = NNPatch(width, height, offset)
            self.w = width;
            self.h = height;
            self.offset = offset;

            self.nndbtr = [];
            self.nndbval = [];
            self.nndbte = [];
        end
    end    
    
    methods (Access = public, Static)
        function nnpatches = GeneratePatch(nndb, sel, h, w, stepx, stepy)
            
        end
    end 
end