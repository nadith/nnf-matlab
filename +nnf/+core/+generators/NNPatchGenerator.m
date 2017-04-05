classdef NNPatchGenerator
    % Generator describes the generator for patches and models.
    %
    % Methods
    % -------
    % generate_patches()
    %     Generates list of NNPatch.
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties (SetAccess = public)
        im_h;
        im_w;
        h;
        w;
        xstride;
        ystride;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        function self = NNPatchGenerator(im_h, im_w, h, w, xstride, ystride) 
            % Constructs a patch generator.
            %
            % Parameters
            % ----------
            % im_h : int
            %     Image height.
            % 
            % im_w : int
            %     Image width.
            % 
            % h : int
            %     Patch height.
            % 
            % w : int
            %     Patch width.
            %
            % xstrip : int
            %     Sliding amount in pixels (x direction).
            % 
            % ystrip : uint16
            %     Sliding amount in pixels (y direction).            
            %   
            
            % Set default values
            if (nargin < 6); ystride=[]; end
            if (nargin < 5); xstride=[]; end
            if (nargin < 4); w=[]; end
            if (nargin < 3); h=[]; end
            if (nargin < 2); im_w=[]; end
            if (nargin < 1); im_h=[]; end
            
            % Imports
            import nnf.db.Format; 
            
            % Set values for instance variables
            self.im_h = im_h;
            self.im_w = im_w;
            self.h = h;
            self.w = w;
            self.xstride = xstride;
            self.ystride = ystride;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nnpatches = generate_nnpatches(self) 
            % Generate the `nnpatches` for the nndb database.
            % 
            % Returns
            % -------
            % list of :obj:`NNPatch`
            %     List of :obj:`NNPatch` instances.
            %
            
            if (isempty(self.im_h) || ...
                isempty(self.im_w) || ...
                isempty(self.h) || ...
                isempty(self.w) || ...
                isempty(self.xstride) || ...
                isempty(self.ystride))
                nnpatch = self.new_nnpatch(self.h, self.w, [1 1]);
                assert(nnpatch.is_holistic);  % Must contain the whole image
                nnpatches = [nnpatch];  % Must be a list
                return;
            end

            % Error handling
            if ((self.xstride ~= 0) && mod((self.im_w - self.w), self.xstride) > 0)
                warning('Patch division will loose some information from the original (x-direction)')
            end

            if ((self.ystride ~= 0) && mod((self.im_h - self.h), self.ystride) > 0)
                warning('WARN: Patch division will loose some information from the original (y-direction)')
            end

            % No. of steps towards x,y direction
            if (self.xstride ~=0)
                x_steps = idivide((self.im_w - self.w), uint16(self.xstride)) + 1;
            else
                x_steps = 1;
            end
            if (self.ystride ~=0) 
                y_steps = idivide((self.im_h - self.h), uint16(self.ystride)) + 1;
            else 
                y_steps = 1;
            end
            
            % Init  variables
            offset = [1 1];
            nnpatches = [];
            
            % Iterate through y direction for patch division
            for i=1:y_steps      
                
                % Iterate through x direction for patch division
                for j=1:x_steps   
                    
                    % Set the patch in nnpatches array
                    nnpatches = [nnpatches self.new_nnpatch(self.h, self.w, offset)];

                    % Update the x direction of the offset
                    offset(2) = offset(2) + self.xstride;
                end
                
                % Update the y direction of the offset, reset x direction offset
                offset(1) = offset(1) + self.ystride;
                offset(2) = 1;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function patch = new_nnpatch(self, h, w, offset)
            % Constructs a new `nnpatch`.
            % 
            % Parameters
            % ----------
            % h : int
            %     Patch height.
            % 
            % w : int
            %     Patch width.
            % 
            % offset : (int, int)
            %     Position of the patch. (Y, X).
            % 
            % Returns
            % -------
            % :obj:`NNPatch`
            %     :obj:`NNPatch` instance.
            % 
            % Note
            % ----
            % Extend this method to construct custom nnpatch.
            %
            
            % Imports
            import nnf.db.NNPatch;            
            patch = NNPatch(h, w, offset);
        end
    end
end



