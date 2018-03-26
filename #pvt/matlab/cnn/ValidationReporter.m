classdef ValidationReporter < nnet.internal.cnn.util.Reporter
    % VALIDATIONREPORTER defines the callbacks for `Trainer.m` reporting.
    %
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    properties
        network    % Live network
        tr_options  % Network's trianingOptions
        ds_val     % Validation datastore
    end
    
    properties
        X_         % Caching variables
        Y_         % Caching variables
    end
        
    methods
        function self = ValidationReporter(tr_options, ds_val)
            % Constructs a `ValidationReporter` object.
            %           
            % Parameters
            % ----------
            % tr_options : object
            %       Network's trianingOptions object.
            %
            % ds_val : obj:`FilePathTableMiniBatchDatasourceEx`
            %       Minibatch datasource for validation data.
            %
            
            self.tr_options = tr_options;
            self.ds_val = ds_val;
            self.X_ = [];
            self.Y_ = [];
        end
        
        function setup( ~ ) 
        end
        
        function start( ~ )
            % Start of Training
        end
        
        function reportIteration(self, summary)
            % Invoke iGatherAndConvert() if summary.(field) is gpuArray
         
            if (mod(summary.Iteration, self.tr_options.ValidationFrequency) == 0 && ~isempty(self.ds_val))                
                
                % if not cached
                if isempty(self.X_) 
                    self.ds_val.reset();
                    self.ds_val.MiniBatchSize = self.ds_val.NumberOfObservations;
                    
                    while (true)
                        try
                            [tmpX, tmpY] = self.ds_val.nextBatch();
                            tmpY = reshape(tmpY, size(tmpY, 3), []);
                            
                            % tmpX is a cell array (1000, 1)
                            for i=1:numel(tmpX)
                                self.X_ = cat(4, self.X_, tmpX{i});
                            end
                            self.Y_ = cat(2, self.Y_, tmpY);
                            
                        catch
                            % All validation data has been visited
                            break;
                        end
                    end
                end
                
                PY = gather(predict(self.network, self.X_));
                PY = reshape(PY, size(PY, 3), []);
                Rvalue = regression(PY, self.Y_, 'one');
                disp(['----------------- RValue:', num2str(Rvalue)])
                % scatter(PY(:), self.Y_(:))
                
                
            end
        end
        
        function reportEpoch( ~, ~, ~, ~ )
            % End of epoch
        end
        
        function finish( ~ )
            % End of training
        end
        
        function computeIteration(self, summary, network)
            self.network = network;            
        end
    end
end

function val = iGatherAndConvert(val)
% Gather if gpuArray and convert to double if numeric
if isnumeric(val)
    val = double( gather( val ) );
end
end

