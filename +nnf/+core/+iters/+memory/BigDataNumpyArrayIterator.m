classdef BigDataNumpyArrayIterator < nnf.core.iters.memory.NumpyArrayIterator
    % BigDataNumpyArrayIterator iterates the raw data in the memory for :obj:`NNModel'.
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
   
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = BigDataNumpyArrayIterator(X, y, nb_class, image_data_pp, params)
            % Construct a :obj:`BigDataNumpyArrayIterator` instance.
            % 
            % Parameters
            % ----------
            % X : `array_like`
            %     Data in 2D matrix. Format: Samples x Features.
            % 
            % y : `array_like`
            %     Vector indicating the class labels.
            % 
            % image_data_pp : :obj:`ImageDataPreProcessor`, sub class of :obj:`ImageDataGenerator`
            %     Image data pre-processor.
            % 
            % params : :obj:`dict`
            %     Core iterator parameters. 
            %

            % Set default values
            if (nargin < 5); params=[]; end          

            self = self@nnf.core.iters.memory.NumpyArrayIterator(X, y, nb_class, image_data_pp, params);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function x = get_data_(self, X, j)
            % Load raw data item from in memory database, pre-process and return.
            % 
            % Parameters
            % ----------
            % X : `array_like`
            %     Data matrix. Format Samples x ...
            % 
            % j : int
            %     Index of the data item to be featched. 
            %
            
            x = self.X(j, :);

            % TODO: Apply necessary transofmraiton
            %%x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x);
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end    
    
    
end


