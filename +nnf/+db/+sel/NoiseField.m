classdef NoiseField < nnf.db.sel.BaseField
    %NOISEFIELD describes the noise definition for Selection.tr_col_indices.
    
    properties (SetAccess = public)
        types;   % Type ('g':gauss, 'c':corruption)
        rates;   % Rate for `tr_col_indices`
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = NoiseField(types, rates)
            % Construct a noise field object.
            % 
            % This field can specify a default value for all tr_col_indices i.e ndarray/scalar, or
            % list of values for each index at Selection.tr_col_indices i.e list of ndarray/scalar.
            % 
            % Parameters
            % ----------
            % types : char or :obj:`list` or string
            %     Type of the noise. 'c': Corruption, 'g': Gaussian
            % 
            % rates : ndarray-double or :obj:`list`
            %     Magnitude of the noise. 0 - 1 range.
            %
            
            self.types = types;
            self.rates = rates;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get_value_at(self, index)
            % Gets the noise field value at the index.
            % 
            % Parameters
            % ----------
            % index: int
            %     The index which the value is required at.
            % 
            % Returns
            % -------
            % value: :obj:`dict`
            %     Noise information available at the index.
            %
            
            % Imports
            import  nnf.db.sel.BaseField;
            
            value.type = BaseField.get_('NoiseField.type', self.types, index);
            value.rate = BaseField.get_('NoiseField.rate', self.rates, index);                       
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function iseq = eq(self, noise_field)
            % Checks for equality of two fields.
            % 
            % Parameters
            % ----------
            % noise_field : :obj:`NoiseField`
            %     Field object to check the equality.
            % 
            % Returns
            % -------
            % bool :
            %     True if equal, False otherwise.
            %
            
            iseq = (isequal(self.rates, noise_field.rates) && ...
                    isequal(self.types, noise_field.types));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end