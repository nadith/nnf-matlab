classdef BaseField
    %BASEFIELD describes the base class for field definitions.
    
    methods (Abstract, Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        value = get_value_at(self, index);  % Get the value at index
        iseq = eq(self, field)         % Checks for eqaulity of two fields
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
           
    methods (Access = protected, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get_(name, values, index, is_cell_field)
            % values = [... ... ...] or <scalar>
            %
            
            % Set defaults for arguments
            if nargin < 4; is_cell_field = false; end
                        
            value = [];
            show_error = false;
            if isempty(index); show_error = true; end
            
            if ~isempty(index) && ~isempty(values)                
                if is_cell_field
                    if iscell(values)
                        if index >= 0 && index <= numel(values) 
                            value = values{index}; 
                        else
                            show_error = true;
                        end                        
                    else % matrix or vector
                        value = values;
                    end
                    
                else
                    if ~isscalar(values)
                        if index >= 0 && index <= numel(values) 
                            value = values(index); 
                        else
                            show_error = true;
                        end                        
                    else % scalar
                        value = values;
                    end
                    
                end
            end
                    
            if show_error
                error([name ': value is not available for the index: ' num2str(index)]);
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
