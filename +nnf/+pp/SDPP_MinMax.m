classdef SDPP_MinMax < nnf.pp.StreamDataPreProcessor
    % SDPP_MINMAX provides min max normalization. 
    
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com). 
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function self = SDPP_MinMax()
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function fit(self, pipeline_id, is_last_data_block, data_block, ulbl)
            % FIT: Fits the `data` for later standardization of the data.
            %   i.e Keep track of the sum and calculate the mean of the data stream.
            %
            % Parameters
            % ----------
            % pipeline_id : int
            %       Unique id of the fit-pipeline.
            %
            % is_last_data_block : bool
            %       Whether it is the last data block.
            %
            % data_block : 2D tensor -double
            %       Data tensor in the format: Features x Samples
            %
            % ulbl : vector -int
            %       Unique label vector for the data block.            
            %                        
            % Returns
            % -------
            % data : 2D tensor -double
            %       Standardized data tensor in the format: Features x Samples
            %
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data_block = standardize(self, data_block, ulbl)            
            % STANDARDIZE: Standardizes the `data`.
            %
            % Parameters
            % ----------
            % data_block : 2D tensor -double
            %       Data tensor in the format: Features x Samples
            %
            % ulbl : vector -int
            %       Unique label vector for the data block.
            %                        
            % Returns
            % -------
            % data_block : 2D tensor -double
            %       Standardized data tensor in the format: Features x Samples
            %
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

