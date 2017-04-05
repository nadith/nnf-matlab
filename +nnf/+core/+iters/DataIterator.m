classdef DataIterator < handle
    % DataIterator represents the base class for all iterators in the
    % Neural Network Framework.
    % 
    % .. warning:: abstract class and must not be instantiated.
    % 
    % Attributes
    % ----------
    % imdata_pp_ : :obj:`ImageDataPreProcessor`
    %     Image data pre-processor for all iterators.
    % 
    % gen_next_ : `function`
    %     Core iterator/generator that provide data.
    % 
    % sync_gen_next_ : :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator`
    %     Core iterator that needs to be synced with `_gen_next`.
    %     (Default value = None)
    % 
    % params_ : :obj:`dict`
    %     Core iterator/generator parameters. (Default value = None)
    % 
    % pp_params_ : :obj:`dict`, optional
    %     Pre-processing parameters for :obj:`ImageDataPreProcessor`.
    %     (Default value = None).
    % 
    % fn_gen_coreiter_ : `function`, optional
    %     Factory method to create the core iterator. (Default value = None).
    % 
    % edataset : :obj:`Dataset`
    %     Dataset enumeration key.
    % 
    % nb_class_ : int
    %     Number of classes.
    % 
    % Notes
    % -----
    % Disman data iterators are utilzing a generator function in _gen_next 
    % while the `nnmodel` data iterators are utilizing an core iterator.
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties (SetAccess = public)
        edataset;
    end
    
    properties (SetAccess = protected)
        imdata_pp_;
        gen_next_;
        sync_gen_next_;
        params_;
        pp_params_;
        fn_gen_coreiter_;        
        nb_class_;
    end
    
    properties (SetAccess = protected, Dependent)
        is_synced;
        sync_gen_next;
        input_vectorized;
        batch_size;
        class_mode;
        nb_sample;        
        nb_class;
        image_shape;
        data_format;
        params;
    end    
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self =  DataIterator(pp_params, fn_gen_coreiter, edataset, nb_class)
            % Constructor of the abstract class :obj:`DataIterator`.
            % 
            % Parameters
            % ----------
            % pp_params : :obj:`dict`
            %     Pre-processing parameters for :obj:`ImageDataPreProcessor`.
            % 
            % fn_gen_coreiter : `function`, optional
            %     Factory method to create the core iterator. (Default value = None). 
            %
            
            % Imports
            import nnf.core.iters.ImageDataPreProcessor;
            
            % Set default values
            if (nargin < 4); nb_class = []; end
            if (nargin < 3); edataset = []; end
            if (nargin < 2); fn_gen_coreiter = []; end
            if (nargin < 1); pp_params = []; end
            
            % Initialize the image data pre-processor with pre-processing params
            % Used by Diskman data iterators and `nnmodel` data iterators
            self.imdata_pp_ = ImageDataPreProcessor(pp_params, fn_gen_coreiter);

            % Core iterator or generator (initilaized in init())
            % Diskman data iterators are utilzing a generator function in _gen_next
            % while the `nnmodel` data iterators are utilizing an core iterator.
            self.gen_next_ = [];

            % :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator`
            % Core iterator that needs to be synced with `_gen_next`.
            self.sync_gen_next_ = [];

            % Iterator params
            % Used by `nnmodel` data iterators only.
            % Can utilize in Diskman data iterators safely in the future.
            self.params_ = [];

            % All the parameters are saved in instance variables to support the 
            % clone() method implementaiton of the child classes.
            self.pp_params_ = pp_params;
            self.fn_gen_coreiter_ = fn_gen_coreiter;

            % Used by `nnmodel` data iterators only.
            self.edataset = edataset;
            self.nb_class_ = nb_class;
        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init(self, gen_next, params)
            % Initialize the instance.
            % 
            % Parameters
            % ----------
            % gen_next_ : `function`
            %     Core iterator/generator that provide data.
            %

            % Set default values
            if (nargin < 2); gen_next = []; end
            if (nargin < 3); params = []; end

            % Set the core iterator or generator 
            self.gen_next_ = gen_next;

            % Set iterator params
            self.params_ = params;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public: Core Iterator Only Operations/Dependant Properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = set_shuffle(self, shuffle)
            % Set shuffle property.

            % This property is only supported by the core iterator
            if (isempty(self.gen_next_)); success = false; return; end

            if (self.gen_next_.batch_index ~= 0 && ... 
                self.gen_next_.shuffle ~= shuffle)
                error('Iterator is already active and failed to set shuffle');
            end

            % Update iterator params if available
            if (~isempty(self.params_))
                self.params_('shuffle') = shuffle;
            end

            self.gen_next_.shuffle = shuffle;
            success = true;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = sync(self, gen_next)
            % Sync the secondary iterator with this iterator.
            % 
            % Sync the secondary core iterator with this core itarator internally.
            % Used when data needs to be generated with its matching target.
            % 
            % Parameters
            % ----------
            % gen : :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator` 
            %     Core iterator that needs to be synced with this core iterator
            %     `gen_next_`.
            % 
            % Note
            % ----
            % Current supports only 1 iterator to be synced.
            %

            % This method is only supported by the core iterator      
            if (isempty(self.gen_next_)); success = false; return; end
            if (isempty(gen_next)); success = false; return; end

            self.gen_next_.sync(gen_next);
            self.sync_gen_next_ = gen_next;
            success = true;
        end   
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = reset(self)
            % Resets the iterator to the begining.
            if (isempty(self.gen_next_)); success = false; return; end
            self.gen_next_.reset();
            success = true;
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Special Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data_batch = next(self)
            % Python iterator interface required method.
            data_batch = self.gen_next_.next();
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Abstract, Access = public)
        [cimg, frecord] = clone(self); % Create a copy of this object.      
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function release_(self)
            % Release internal resources used by the iterator.
            self.imdata_pp_ = [];
            self.gen_next_ = [];
            self.sync_gen_next_ = [];
            self.params_ = [];
            delete(self.pp_params_);
            self.fn_gen_coreiter_ = [];
            self.edataset = [];
            self.nb_class_ = [];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
     
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Dependant Properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.is_synced(self)
            % bool : whether this generator is synced with another generator.
            value = (~isempty(self.sync_gen_next_));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.sync_gen_next(self)
            % bool : whether this generator is synced with another generator.
            value = self.sync_gen_next;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function value = get.input_vectorized(self)
            % bool: whether the input needs to be vectorized via the core iterator.
            % This property is only supported by the core iterator
            if (isempty(self.gen_next_)); value = false; return; end
            
            value = self.gen_next_.input_vectorized;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function value = get.batch_size(self)
            % int: batch size to be read by the core iterator.
            % This property is only supported by the core iterator
            if (isempty(self.gen_next_)); value = false; return; end
            
            value = self.gen_next_.batch_size;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function value = get.class_mode(self)
            % str: class mode at core iterator.
            % This property is only supported by the core iterator
            if (isempty(self.gen_next_)); value = false; return; end

            value = self.gen_next_.class_mode;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function value = get.nb_sample(self)
            % int: number of samples registered at core iterator/generator.
            % This property is only supported by the core iterator
            if (isempty(self.gen_next_)); value = false; return; end

            value = self.gen_next_.nb_sample;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function value = get.nb_class(self)
            % int: number of classes registered at core iterator/generator.
            % This property is only supported by the core iterator
            if (isempty(self.gen_next_)); value = false; return; end
            
            value = self.gen_next_.nb_class;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function value = get.image_shape(self)
            % :obj:`tuple` : shape of the image that is natively producted by this iterator.
            % This property is only supported by the core iterator      
            if (isempty(self.gen_next_)); value = false; return; end

            value = self.gen_next_.image_shape;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.data_format(self)
            % :obj:`tuple` : shape of the image that is natively producted by this iterator.
            % This property is only supported by the core iterator      
            if (isempty(self.gen_next_)); value = false; return; end
            
            value = self.gen_next_.data_format;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.params(self)
            % :obj:`dict`: Core iterator parameters.
            % Imports
            import nnf.utl.Map;
            
            if (isempty(self.params_)); value = Map(); return; end
            value = self.params_;
        end
    end
end    
        


