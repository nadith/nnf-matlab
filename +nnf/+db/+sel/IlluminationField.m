classdef IlluminationField < nnf.db.sel.BaseField
    %ILLUMINATIONFIELD describes the illumination definition for Selection.tr_col_indices.
    
    properties (SetAccess = public)
        positions;      % {[33 33], [15 10], [5 10], [15 5]};
        covariances;    % {[1 0; 0 1] [1 0; 0 1] [1 0; 0 1] [1 0; 0 1]}        
        brightness;     % [0.5 0.3 0.5 0.4 0.5];    % 0 - 1 range
        darkness;       % [0.1 0.3 0.5 1 0.5];      % 0 - 1 range
        thresholds;     % [0.0039 0.1 0.2 0.5 0.5]; % 1/255
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = IlluminationField(positions, covariances, brightness, darkness, thresholds)
            % Construct a illumination field object.
            % 
            % This field can specify a default value for all tr_col_indices i.e ndarray/scalar, or
            % list of values for each index at Selection.tr_col_indices i.e list of ndarray/scalar.
            % 
            % Parameters
            % ----------
            % positions : ndarray-int or :obj:`list`
            %     Position to place the light source for all tr_col_indices. i.e ndarray([33, 33])
            % 
            % covariances : ndarray-double or :obj:`list`
            %     Covariance matrix to define the spread of the light source. i.e ndarray([[1, 0], [0, 1]])
            % 
            % brightness : ndarray-double or :obj:`list`, optional, default: 1
            %     The factor of brightness in illumination. 0 - 1 range.
            % 
            % darkness : ndarray-double or :obj:`list`, optional, default: 1
            %     The factor of darkness to darken the non-illuminated region. 0 - 1 range.
            % 
            % thresholds : ndarray-double or :obj:`list`, optional, default: 1/255
            %     The threshold to define the illumination mask in the generated gaussian light source. 0 - 1 range.
            %     1/255 thresholds the illuminated region where a 1 pixel shift is expected after applying
            %     the generated gaussian noise.
            %
            
            % Set defaults for arguments
            % Capture when a pixel value is shifted by 1 due to the light source
            if nargin < 5; thresholds = 1/255; end 
            if nargin < 4; darkness = 1; end    % Full darkness factor
            if nargin < 3; brightness = 1; end  % Full brightness factor
            
            self.positions = positions;
            self.covariances = covariances;
            self.brightness = brightness;
            self.darkness = darkness;
            self.thresholds = thresholds;
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get_value_at(self, index)
            % Gets the Occlusion field value at the index.
            % 
            % Parameters
            % ----------
            % index: int
            %     The index which the value is required at.
            % 
            % Returns
            % -------
            % value: :obj:`dict`
            %     Occlusion information available at the index.
            %
            
            % Imports
            import  nnf.db.sel.BaseField;
            
            value.position = BaseField.get_('IlluminationField.positions', self.positions, index, true);
            value.covariance = BaseField.get_('IlluminationField.covariances', self.covariances, index, true);
            value.brightness = BaseField.get_('IlluminationField.brightness', self.brightness, index);
            value.darkness = BaseField.get_('IlluminationField.darkness', self.darkness, index);
            value.threshold = BaseField.get_('IlluminationField.threshold', self.thresholds, index);
        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function iseq = eq(self, illu_field)
            % Checks for equality of two fields.
            % 
            % Parameters
            % ----------
            % illu_field : :obj:`IlluminationField`
            %     Field object to check the equality.
            % 
            % Returns
            % -------
            % bool :
            %     True if equal, False otherwise.
            %
            
            iseq = (isequal(self.covariances, illu_field.covariances) && ...
                    isequal(self.positions, illu_field.positions) && ...
                    isequal(self.brightness, illu_field.brightness) && ...
                    isequal(self.darkness, illu_field.darkness) && ...
                    isequal(self.thresholds, illu_field.thresholds));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
