classdef MemDataIterator < nnf.core.iters.DataIterator
    % MemDataIterator iterates the data in memory for :obj:`NNModel'.
    % 
    % Attributes
    % ----------
    % nndb : :obj:`NNdb`
    %     Database to iterate.
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
        nndb;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = MemDataIterator(edataset, nndb, nb_class, pp_params, fn_gen_coreiter)
            % Construct a MemDataIterator instance.
            % 
            % Parameters
            % ----------
            % edataset : :obj:`Dataset`
            %     Dataset enumeration key.
            % 
            % nndb : :obj:`NNdb`
            %     In memory database to iterate.
            % 
            % nb_class : int
            %     Number of classes.
            % 
            % pp_params : :obj:`dict`, optional
            %     Pre-processing parameters for :obj:`ImageDataPreProcessor`. 
            %     (Default value = None). 
            % 
            % fn_gen_coreiter : `function`, optional
            %     Factory method to create the core iterator.
            %     (Default value = None).
            %
            
            % Set default values
            if (nargin < 5); fn_gen_coreiter = []; end
            if (nargin < 4); pp_params = []; end

            self = self@nnf.core.iters.DataIterator(pp_params, fn_gen_coreiter, edataset, nb_class);

            % NNdb database to create the iterator
            self.nndb = nndb;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function settings = init_ex(self, params, y, setting)
            % Initialize the instance.
            % 
            % Parameters
            % ----------
            % params : :obj:`dict`
            %     Core iterator parameters.

            % Imports
            import nnf.utl.Map;
            import nnf.core.K;

            % Set default values
            if (nargin < 4); setting = []; end
            if (nargin < 3); y = []; end
            if (isempty(params)); params = Map(); end       

            % Set db and _image_shape (important for convolutional nets, and fit() method below)
            if (params.isKey('data_format'))
                data_format = params.get('data_format');
            else
                data_format = K.image_data_format;
            end        
            target_size = [self.nndb.h self.nndb.w];

            db = [];
            if (self.nndb.ch == 3)  % 'rgb'
                if (strcmp(data_format, 'channels_last'))
                    params('_image_shape') = [target_size 3];
                    db = self.nndb.db_convo_tf;
                else
                    params('_image_shape') = [3 target_size];
                    db = self.nndb.db_convo_th;
                end

            else
                if (strcmp(data_format, 'channels_last'))
                    params('_image_shape') = [target_size 1];
                    db = self.nndb.db_convo_tf;
                else
                    params('_image_shape') = [1 target_size];
                    db = self.nndb.db_convo_th;
                end
            end

            % Required for featurewise_center, featurewise_std_normalization and 
            % zca_whitening. Currently supported only for in memory datasets.
            if (self.imdata_pp_.featurewise_center || ...
                self.imdata_pp_.featurewise_std_normalization || ...
                self.imdata_pp_.zca_whitening || ...
                (~isempty(self.pp_params_) && self.pp_params_.isKey('mapminmax')))
                self.imdata_pp_.fit(db, ...
                                    self.imdata_pp_.augment, ...
                                    self.imdata_pp_.rounds, ...
                                    self.imdata_pp_.seed);
            else
                self.imdata_pp_.apply(setting);
            end

            gen_next = self.imdata_pp_.flow_ex(db, self.nndb.cls_lbl', self.nb_class, params);
            settings = self.imdata_pp_.settings;
            self.init(gen_next, params);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function new_obj = clone(self)
            % Imports
            import nnf.core.iters.memory.MemDataIterator;

            % Create a copy of this object.
            new_obj = MemDataIterator(self.edataset, self.nndb, self.nb_class, ...
                                        self.pp_params_, self.fn_gen_coreiter_);
            new_obj.init_ex(self.params_);
            new_obj.sync(self.sync_gen_next_);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = sync_generator(self, iter)
            % Sync the secondary iterator with this iterator.
            % 
            % Sync the secondary core iterator with this core iterator internally.
            % Used when data needs to be generated with its matching target.
            % 
            % Parameters
            % ----------
            % iter : :obj:`MemDataIterator`
            %     Iterator that needs to be synced with this iterator.
            % 
            % Note
            % ----
            % Current supports only 1 iterator to be synced.
            %        
            if (isempty(iter)); success = false; return; end
            self.sync(iter.gen_next_);
            success = true;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function release(self)
            % Release internal resources used by the iterator.
            self = release@nnf.core.iters.DataIterator();
            self.nndb = [];    
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
