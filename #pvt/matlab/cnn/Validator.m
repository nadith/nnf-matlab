classdef Validator < handle
    % TODO: [UNUSED] This class is not used since `validate(self, info)`: info object does not the pass network
    % object to perform validation
    %
    % VALIDATOR provides the context for validate() method.
    %   Pass validate() method handle to 'OutputFcn' argument when defining trainigOptions for
    %   network.
    %
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    properties
        name        % Name of the `Validator` object.
        trOptions   % Network's trianingOptions
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = Validator(name)
            % Constructs a `Validator` object.
            %           
            % Parameters
            % ----------
            % name : string
            %       Name of the `Validator` object.
            %
            % options : object
            %       Network's trianingOptions object
            %
            self.name = name;
            self.trOptions = [];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = init(self, trOptions)
            % Initializes `self` object with trainingOptions `options`.
            %           
            % Parameters
            % ----------
            % options : object
            %       Network's trianingOptions object
            %
            self.trOptions = trOptions;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function stop = validate(self, info)
            % Pass this method handle to 'OutputFcn' argument when defining trainigOptions for
            % network.
            stop = false;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end