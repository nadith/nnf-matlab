classdef SDPP_ZScore < nnf.pp.StreamDataPreProcessor
    % SDPP_ZSCORE provides zscore normalization.
    
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com). 
    
    properties (SetAccess = public)
        mean__;         % Mean
        std__;          % Mean
        n__;            % Sample count
        csum__;         % Cumilated sum
        csum_diff__;    % Cumilated sum of differences (used for std calculation)        
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function self = SDPP_ZScore()
            self.mean__ = [];
            self.std__ = [];
            self.n__ = 0;
            self.csum__ = 0;
            self.csum_diff__ = 0;            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = fit(self, pipeline_id, is_last_data_block, data_block, ulbl)
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
                        
            if isempty(self.mean__)
                self.csum__ = self.csum__ + sum(data_block, 2);
                self.n__ = self.n__ + size(data_block, 2);
            else                
                self.csum_diff__ = sum((data_block - self.mean__).^2, 2);
            end
            
            if (is_last_data_block)                
                if isempty(self.mean__)
                    self.mean__ = self.csum__ / self.n__;                    
                else
                    self.std__  = sqrt(self.csum_diff__ / self.n__);                    
                end
            end
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
            
            data_block = (data_block - self.mean__) / self.std__;            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

