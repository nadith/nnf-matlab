classdef OcclusionField < nnf.db.sel.BaseField
    %OCCLUSIONFIELD describes the occlusion definition for Selection.tr_col_indices.
    
    properties (SetAccess = public)
        types;   % Type ('t':top, 'b':bottom, 'l':left, 'r':right, 'v':vertical, 'h':horizontal)
        rates;   % Rate for `tr_col_indices`        
        offsets; % Start offset from top/bottom/left/right corner for specified type
        filters; % User-defined occlusion patterns for `tr_col_indices`
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = OcclusionField(types, rates, offsets, filters)
            % Construct a occlusion field object.
            % 
            % This field can specify a default value for all tr_col_indices i.e ndarray/scalar, or
            % list of values for each index at Selection.tr_col_indices i.e list of ndarray/scalar.
            % 
            % Parameters
            % ----------
            % types : char
            %     Type of the occlusion. 'b': Bottom, 't': Top, 'l': left, 'r': Right, 'v': Vertical, 'h': Horizontal.
            % 
            % rates : ndarray-double
            %     Magnitude of the occlusion. 0 - 1 range.
            % 
            % offsets : ndarray-int, optional, default: None
            %     Start offset from top/bottom/left/right corner for the specified type.
            %     If None, the offset defaults to 0.
            % 
            % filters : ndarray-double, optional, default: None
            %     User defined occlusion pattern.
            %
            
            % Set defaults for arguments
            if nargin < 4; filters = []; end
            if nargin < 3; offsets = []; end
           
            % Initialize the fields
            self.types = types;
            self.rates = rates;            
            self.offsets = offsets;
            self.filters = filters;
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
            
            value.type = BaseField.get_('OcclusionField.type', self.types, index);
            value.rate = BaseField.get_('OcclusionField.rate', self.rates, index);
            value.offset = BaseField.get_('OcclusionField.offset', self.offsets, index);
            value.filter = BaseField.get_('OcclusionField.filter', self.filters, index, true);
            
            % No offset
            if isempty(value.offset); value.offset = 0; end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function iseq = eq(self, occl_field)
            % Checks for equality of two fields.
            % 
            % Parameters
            % ----------
            % occl_field : :obj:`OcclusionField`
            %     Field object to check the equality.
            % 
            % Returns
            % -------
            % bool :
            %     True if equal, False otherwise.
            %
            
            iseq = (isequal(self.rates, occl_field.rates) && ...
                    isequal(self.types, occl_field.types) && ...
                    isequal(self.offsets, occl_field.offsets) && ...
                    isequal(self.filters, occl_field.filters));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end