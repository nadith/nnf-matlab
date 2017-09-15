classdef ImageDataPreProcessor < nnf.core.iters.ImageDataGenerator
    % ImageDataPreProcessor represents the pre-processor for image data.
    % 
    % Attributes
    % ----------
    % _fn_gen_coreiter : `function`, optional
    %     Factory method to create the core iterator. (Default value = None).
    % 
    % _nrm_vgg16 : bool, optional
    %     Whether to perform unique normalization for VGG16Model. (Default value = False).
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties (SetAccess = protected)
        fn_gen_coreiter_;
        pp_params_;
        mapminmax_setting_;
    end
    
    properties (SetAccess = protected, Dependent)
        settings;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = ImageDataPreProcessor(pp_params, fn_gen_coreiter)
            % Construct a :obj:`ImageDataPreProcessor` instance.
            % 
            % Parameters
            % ----------
            % pp_params : :obj:`dict`, optional
            %     Pre-processing parameters. (Default value = None).
            % 
            % fn_gen_coreiter : `function`, optional
            %     Factory method to create the core iterator. (Default value = None). 
            %
            
            % Set default values
            if (nargin < 2); fn_gen_coreiter = []; end
            if (nargin < 1); pp_params = []; end
            
            self = self@nnf.core.iters.ImageDataGenerator(pp_params);            
            self.fn_gen_coreiter_ = fn_gen_coreiter;
            self.pp_params_ = pp_params;
            self.mapminmax_setting_ = [];            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function apply(self, settings)
            % Apply settings on `self`.
            
            if (isempty(settings)); return; end
            self.mean = settings.mean;
            self.std = settings.std;
            self.principal_components = settings.principal_components;
            self.mapminmax_setting_ = settings.mapminmax_setting_;
            self.whiten = settings.whiten;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function x = standardize(self, x)
            % Standardize data sample.
            
            % New custom pre-processing param
            % VGG16Model specific pre-processing param
            if (~isempty(self.pp_params_) && ...
                    self.pp_params_.isKey('normalize_vgg16') && ...
                    self.pp_params_.get('normalize_vgg16')) 
                x(0, :, :) = x(0, :, :) - 103.939;
                x(1, :, :) = x(1, :, :) - 116.779;
                x(2, :, :) = x(2, :, :) - 123.68;
            end
            
            x = standardize@nnf.core.iters.ImageDataGenerator(self, x);
            
            % Map min max normalization
            if (~isempty(self.mapminmax_setting_))
                x = mapminmax('apply', x', self.mapminmax_setting_)';
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function fit(self, X, augment, rounds, seed)
            
            % Set default values
            if (nargin < 5); seed = []; end
            if (nargin < 4); rounds = 1; end
            if (nargin < 3); augment = false; end
            
            fit@nnf.core.iters.ImageDataGenerator(self, X, augment, rounds, seed)
            
            % Perform whitening/mapminmax/etc
            if (~isempty(self.pp_params_) && ...
                    self.pp_params_.isKey('mapminmax'))
                    minmax_range = self.pp_params_.get('mapminmax');
                [~, self.mapminmax_setting_] = mapminmax(X', minmax_range(1), minmax_range(2)); 
            end
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function core_iter = flow(self, X, y, nb_class, params)
            % Construct a core iterator instancef for in memory database traversal.
            % 
            % Parameters
            % ----------
            % X : `array_like`
            %     Data in tensor. Format: Samples x dim1 x dim2 ...
            % 
            % y : `array_like`
            %     Vector indicating the class labels.
            % 
            % params : :obj:`dict`
            %     Core iterator parameters. 
            % 
            % Returns
            % -------
            % :obj:`NumpyArrayIterator`
            %     Core iterator.
            %

            % Imports
            import nnf.core.iters.memory.NumpyArrayIterator;

            % Set default values
            if (nargin < 5); params = []; end
            if (nargin < 4); nb_class = []; end
            if (nargin < 3); y = []; end

            if (isempty(self.fn_gen_coreiter_))
                core_iter = NumpyArrayIterator(X, y, nb_class, self, params);
                return;

            else
                try
                    core_iter = self.fn_gen_coreiter_(X, y, nb_class, self, params);
                    if (~isa(core_iter, 'NumpyArrayIterator'))
                        error('`fn_gen_coreiter_` is not a child of `NumpyArrayIterator` class');
                    end

                catch ME
                    error(sprintf('%s', ME.identifier));
                end
            end
        end     
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        % function core_iter = flow_from_directory(self, frecords, nb_class, params)
        %     % Construct a core iterator instancef for disk database traversal.
        %     % 
        %     % Parameters
        %     % ----------
        %     % frecords : :obj:`list`
        %     %     List of file records. frecord = [fpath, fpos, cls_lbl] 
        %     % 
        %     % nb_class : int
        %     %     Number of classes.
        %     % 
        %     % params : :obj:`dict`
        %     %     Core iterator parameters. 
        %     % 
        %     % Returns
        %     % -------
        %     % :obj:`DirectoryIterator`
        %     %     Core iterator.
        %     %
        % 
        %     % Imports
        %     import nnf.core.iters.disk.DirectoryIterator;
        % 
        %     % Set default values
        %     if (nargin < 4); params = []; end
        % 
        %     if (isempty(self.fn_gen_coreiter_))
        %         core_iter = DirectoryIterator(frecords, nb_class, self, params);
        %         return;            
        % 
        %     else
        %         try
        %             core_iter = self.fn_gen_coreiter_(frecords, nb_class, self, params)
        %             if (~isa(core_iter, 'DirectoryIterator'))
        %                 error('`fn_gen_coreiter_` is not a child of `DirectoryIterator` class');
        %             end
        % 
        %         catch ME
        %             error(sprintf('%s', ME.identifier));
        %         end
        %     end
        % end
    end   
        
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Dependant Properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.settings(self)
            % Calculated settings to use on val/te/etc datasets.
            value.mean = self.mean;
            value.std = self.std;
            value.principal_components = self.principal_components;
            value.mapminmax_setting_ = self.mapminmax_setting_;
            value.whiten = self.whiten;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end