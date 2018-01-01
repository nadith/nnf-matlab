classdef CNNNet < handle
    % CNNNet Base class for Matlab inbuilt CNN networks.
    
    properties (SetAccess = public)
        layers;         % Layer definitions
        tr_options;     % Training options
    end
    
    properties (SetAccess = protected)
        params_;        % varargs provided @ the constructor.
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = CNNNet(tr_options, layerDefinitions, varargin)
            % Constructs a `CNNNet` object.
            %            
            % Parameters
            % ----------
            % tr_options : struct
            %       Training options structure resulting from invoking `trainingOptions(...)`.            
            %
            % layersTransfer : cell array
            %       Each cell indicating layers for the CNN network.
            %
            % varargin : 
            %       UseAlexNet : bool       - If pretrained model `AlexNet` need to be used.
            %       NumberOfClasses : int   - For classification problems. (Only used for pretrained models)
            %       OutputNodes : int       - For regression problems. (Only used for pretrained models)
            %
            
            % Set defaults for arguments
            if (nargin < 1)
                tr_options = [];
            end
            
            if (nargin < 2)
                layerDefinitions = [];
            end
                                    
            p = inputParser;            
            defaultUseAlexNet = false;
            defaultNumberOfClasses = 0;
            defaultOutputNodes = 0;
            addParameter(p, 'UseAlexNet', defaultUseAlexNet);
            addParameter(p, 'NumberOfClasses', defaultNumberOfClasses);
            addParameter(p, 'OutputNodes', defaultOutputNodes);
            parse(p, varargin{:});
            self.params_ = p.Results;
            
            % Error handling
            if isempty(tr_options)
                error(['Training options should be provided']);
            end
            
            if (isempty(layerDefinitions) && ~self.params_.UseAlexNet)
                error(['Invalid layer definitions or no pre-trained model is chosen']);
            end
            
            if (~isempty(layerDefinitions) && self.params_.UseAlexNet)
                error(['Use either layer definitions or pre-trained model']);
            end
            
            if (~isempty(layerDefinitions) && self.params_.NumberOfClasses > 0)
                warning(['`NumberOfClasses` will be ignored due to the definition of the layers']);
            end
            
            if (~isempty(layerDefinitions) && self.params_.OutputNodes > 0)
                warning(['`OutputNodes` will be ignored due to the definition of the layers']);
            end
            
            if (self.params_.UseAlexNet && self.params_.NumberOfClasses > 0 && self.params_.OutputNodes > 0)
                error(['`OutputNodes` and `NumberOfClasses` both cannot be defined at once. Use one of them']);
            end
            
            if (self.params_.UseAlexNet && self.params_.NumberOfClasses == 0 && self.params_.OutputNodes == 0)
                error(['One of the quantities; `OutputNodes` or `NumberOfClasses` must be defined']);
            end            
            
            % Initialize training options
            self.init(tr_options);
            
            % If the network definition is given, 
            if ~isempty(layerDefinitions)
                self.layers = layerDefinitions;
                
            elseif (self.params_.UseAlexNet)
                
                net = alexnet;
                self.layers = net.Layers(1:end-3);
                
                % For classification problems
                if (self.params_.NumberOfClasses > 0)
                    self.layers = [
                        self.layers
                        fullyConnectedLayer(self.params_.NumberOfClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
                        softmaxLayer
                        classificationLayer];
                else                    
                    self.layers = [
                        self.layers
                        fullyConnectedLayer(self.params_.OutputNodes, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
                        regressionLayer];
                end                
            end            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init(self, tr_options)
            self.tr_options = tr_options;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [net, XX, XXT] = train_datagen(self, data_gen)
            % Training with a data generator with next() function.
            %   Refer: `nnf.pp.MovingWindowDataGenerator`
            
            XX = [];
            XXT = [];
            for wsize = 6:2:10
                data_gen.init((wsize/2), wsize); % (3,6) , (4,8) , (5,10) , ...

                while ~(data_gen.IsEnd)
                    [X, XT] = data_gen.next();

                    % Convert it to 3 channels if X is 2D
                    if (numel(size(X)) == 2)
                        X = repmat(X, [1 1 3]);
                    end                       
                        
                    % Resize
                    X = imresize(X, [227 227]);                    
                    XX = cat(4, XX, X(:, :, :, 1));
                    XXT = cat(1, XXT, XT);
                end
            end
                    
            net = trainNetwork(XX, XXT, self.layers, self.tr_options);
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function net = train(self, XX, XXT)
            % Training with in memory dataset
            net = trainNetwork(XX, XXT, self.layers, self.tr_options);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function net = train_datasource(self, datasource, reporter)
            % Training with a data source
            if (nargin < 3)
                reporter = [];
            end
            net = trainNetworkEx(datasource, self.layers, self.tr_options, reporter);
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function net = train_table(self, tbl)
            % Training with a table
            net = trainNetwork(tbl, self.layers, self.tr_options);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

