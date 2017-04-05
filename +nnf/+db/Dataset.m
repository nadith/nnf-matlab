classdef Dataset < uint32
    %DATASET Summary of this class goes here
    %   Detailed explanation goes here
    
    enumeration
        % Do not change the order of the values
        % Refer NNDiskMan.process() for more details
        TR      (1) 
        VAL     (2)
        TE      (3)
        TR_OUT  (4)
        VAL_OUT (5)
        TE_OUT  (6)
    end
    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function eval = enum(int_value)
            % Imports
            import nnf.db.Dataset;
            
            if (int_value == 1)
                eval = Dataset.TR;

            elseif (int_value == 2)
                eval = Dataset.VAL;

            elseif (int_value == 3)
                eval = Dataset.TE;

            elseif (int_value == 4)
                eval = Dataset.TR_OUT;

            elseif (int_value == 5)
                eval = Dataset.VAL_OUT;
                
            elseif (int_value == 6)
                eval = Dataset.TE_OUT;

            else
                error('Unsupported');
                
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function  str = str(edataset)
            % Imports
            import nnf.db.Dataset;
            
            if (edataset == Dataset.TR)
                str = 'Training';

            elseif (edataset == Dataset.VAL)
                str = 'Validation';

            elseif (edataset == Dataset.TE)
                str = 'Testing';

            elseif (edataset == Dataset.TR_OUT)
                str = 'TrTarget';

            elseif (edataset == Dataset.VAL_OUT)
                str = 'ValTarget';

            elseif (edataset == Dataset.TE_OUT)
                str = 'TeTarget';

            else
                error('Unsupported');
                
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function elist = get_enum_list()
            % Imports
            import nnf.db.Dataset;  
            
            elist = [Dataset.TR Dataset.VAL Dataset.TE Dataset.TR_OUT Dataset.VAL_OUT Dataset.TE_OUT];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

end