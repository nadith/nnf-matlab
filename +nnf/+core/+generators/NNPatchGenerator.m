classdef NNPatchGenerator
    % Generator describes the generator for patches and models.
    %
    % Methods
    % -------
    % generate_patches()
    %     Generates list of NNPatch.
    
    properties
        im_h
        im_w
        h
        w
        xstrip
        ystrip
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        function self = NNPatchGenerator(im_h, im_w, pat_h, pat_w, xstrip, ystrip) 
            % Constructs a patch generator.
            %
            % Parameters
            % ----------
            % im_h : uint16
            %     Image height.
            %
            % im_w : uint16
            %     Image width.
            %
            % pat_h : uint16
            %     Height of the patch.
            % 
            % pat_w : uint16
            %     Width of the patch.
            %
            % xstrip : uint16
            %     Sliding amount in pixels (x direction).
            % 
            % ystrip : uint16
            %     Sliding amount in pixels (y direction).            
            %
            
            % Imports
            import nnf.db.Format; 
            
            % Set values for instance variables
            self.im_h = im_h;
            self.im_w = im_w;
            self.h = pat_h;
            self.w = pat_w;
            self.xstrip = xstrip;
            self.ystrip = ystrip;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nnpatches = generate_patches(self) 
            % Generate the patches for the nndb database.
            % 
            % Returns
            % -------
            % nnpatches : list- NNPatch
            %     Patch objects.
                           
            % Error handling
            if (mod((self.im_w - self.w), self.xstrip) > 0) 
                warning('Patch division will loose some information from the original (x-direction)');
            end                
            if (mod((self.im_h - self.h), self.ystrip) > 0)
                warning('Patch division will loose some information from the original (y-direction)');           
            end
            
            % No. of steps towards x,y direction
            x_steps = idivide((self.im_w - self.w), uint16(self.xstrip)) + 1;
            y_steps = idivide((self.im_h - self.h), uint16(self.ystrip)) + 1;
            
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
                    offset(2) = offset(2) + self.xstrip;
                end
                
                % Update the y direction of the offset, reset x direction offset
                offset(1) = offset(1) + self.ystrip;
                offset(2) = 1;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function patch = new_nnpatch(self, h, w, offset)
            % Overridable
            
            % Imports
            import nnf.db.NNPatch; 
            
            patch = NNPatch(h, w, offset);
        end
    end
end



