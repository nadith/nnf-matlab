classdef NumpyArrayIterator < nnf.core.iters.Iterator
    % NumpyArrayIterator iterates the image data in the memory for :obj:`NNModel'.
    % 
    % Attributes
    % ----------
    % input_vectorized : bool
    %     Whether the data needs to be returned via the iterator as a batch of data vectors.
    %
    
    properties
        input_vectorized;
        X;
        y;
        image_data_generator;
        data_format;
        image_shape;
        class_mode;
        save_to_dir;
        save_prefix;
        save_format;
        nb_sample;
        nb_class;
        sync_gen;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = NumpyArrayIterator(X, y, nb_class, imdata_pp, params)
            % Construct a :obj:`NumpyArrayIterator` instance.
            % 
            % Parameters
            % ----------
            % X : `array_like`
            %     Data in 2D matrix. Format: Samples x Features.
            % 
            % y : `array_like`
            %     Vector indicating the class labels.
            % 
            % imdata_pp : :obj:`ImageDataPreProcessor`
            %     Image data pre-processor.
            % 
            % params : :obj:`dict`
            %     Core iterator parameters. 
            %         
            
            % Imports 
            import nnf.core.K;
            
            % Set default values
            if (nargin < 5); params = []; end
            
            % Set defaults for params
            size_X = size(X); image_shape = size_X(1:end);
        
            if (isempty(params))
                input_vectorized = false;
                data_format = [];
                class_mode = '';
                batch_size = 32; shuffle = true; seed = [];
                save_to_dir = []; save_prefix = ''; save_format = 'jpeg'

            else
                if (params.isKey('input_vectorized')); input_vectorized = params.get('input_vectorized'); else; input_vectorized = false; end
                image_shape = params.setdefault('_image_shape', image_shape); % internal use
                if (params.isKey('data_format')); data_format = params.get('data_format'); else; data_format = []; end
                if (params.isKey('class_mode')); class_mode = params.get('class_mode'); else; class_mode = ''; end
                if (params.isKey('batch_size')); batch_size = params.get('batch_size'); else; batch_size = 32; end
                if (params.isKey('shuffle')); shuffle = params.get('shuffle'); else; shuffle = true; end
                if (params.isKey('seed')); seed = params.get('seed'); else; seed = []; end
                if (params.isKey('save_to_dir')); save_to_dir = params.get('save_to_dir'); else; save_to_dir = []; end
                if (params.isKey('save_prefix')); save_prefix = params.get('save_prefix'); else; save_prefix = ''; end
                if (params.isKey('save_format')); save_format = params.get('save_format'); else; save_format = 'jpeg'; end
            end
            
            if (~isempty(y) && (size(X, 1) ~= size(y, 1)))
                error(sprintf('X (images tensor) and y (labels) should have the same length. Found: X.shape = %s, y.shape = %s', ...
                        num2str(size(X)), num2str(size(y))));
            end
            
            if isempty(data_format)
                data_format = K.image_data_format();
            end
            
            % [LIMITATION: PYTHON-MATLAB]
            % N x H x W x CH=1 or N x 1 x H x W=1 (last singleton index is not supported in matlab)
            % https://au.mathworks.com/matlabcentral/answers/21992-singleton-dimention-as-last-dimension-in-matrix
            if (ndims(X) < 2)
                error(['Input data in `NumpyArrayIterator` should have rank >= 2. ' ...
                    'You passed an array with shape ' num2str(size_X)]);
                
            elseif (ndims(X) >= 3)
                if (strcmp(data_format, 'channels_last')); channels_axis = 4; else; channels_axis = 2; end            
                if (ndims(X) == 3); chs = 1; else; chs = size_X(channels_axis); end            
                if ~(chs == 1 || chs == 3 || chs == 4)
                    error(['NumpyArrayIterator is set to use the '...
                                     'data format convention "' data_format '" '...
                                     '(channels on axis ' num2str(channels_axis) '), i.e. expected '...
                                     'either 1, 3 or 4 channels on axis ' num2str(channels_axis) '. '...
                                     'However, it was passed an array with shape ' num2str(size_X) ...
                                     ' (' num2str(chs) ' channels).']);
                end
                
            else % if (ndims(X) == 2) 
                % X via `BigDataNumpyArrayIterator` is N x H x 1 x 1 (but last singleton index is not supported in matlab)
                
            end
                        
            % [LIMITATION: PYTHON-MATLAB]
            % Error: A constructor call to superclass nnf.core.iters.Iterator appears after the object is used, or after a return            
            nb_sample =  size(X, 1);
            self = self@nnf.core.iters.Iterator(nb_sample, batch_size, shuffle, seed);
            
            % NO NEED
            % if y is not None:
            %     self.y = np.asarray(y)
            % else:
            %     self.y = None
            self.X = X; % np.asarray(X); NO NEED
            self.y = y;
            
            self.input_vectorized = input_vectorized;
            self.image_data_generator = imdata_pp;
            self.data_format = data_format;
            self.image_shape = image_shape;
            
            if (~ismember(class_mode, {'categorical', 'binary', 'sparse', ''}))
                error(['Invalid class_mode:' class_mode ...
                                 '; expected one of "categorical", '...
                                 '"binary", "sparse", or None.']);
            end
            
            self.class_mode = class_mode;
            self.save_to_dir = save_to_dir;
            self.save_prefix = save_prefix;
            self.save_format = save_format;
            self.nb_sample = nb_sample;   
            self.nb_class = nb_class;
            self.sync_gen = [];  % Synced generator            
            
            % [LIMITATION: PYTHON-MATLAB]
            % Error: A constructor call to superclass nnf.core.iters.Iterator appears after the object is used, or after a return
            % self = self@nnf.core.iters.Iterator(self.nb_sample, batch_size, shuffle, seed);
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function sync(self, gen)
            % Both generators must have same number of images
            assert(size(self.X, 1) == size(gen.X, 1))
            if (self.batch_size ~= gen.batch_size)
                warning('`batch_size` of synced generators are different.');
            end

            self.sync_gen = gen;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data_batch = next(self)
            % Advance to next batch of data.
            % 
            % Returns
            % -------
            % `array_like` :
            %     `batch_x` data matrix. Format: Samples x Features
            % 
            % `array_like` :
            %     `batch_y` class label vector or 'batch_x' matrix. refer code.
            %
            
            % for python 2.x.
            % Keeps under lock only the mechanism which advances
            % the indexing of each batch
            % see http://anandology.com/blog/using-iterators-and-generators/
            % with self.lock: NO NEED
                [index_array, current_index, current_batch_size] = self.flow_index();
            % The transformation of images is not under thread lock so it can be done in parallel            
            batch_x = zeros([current_batch_size self.image_shape]);
            
            sync_gen = self.sync_gen;
            batch_xt = [];        
            if (~isempty(sync_gen))
                batch_xt = zeros([current_batch_size self.sync_gen.image_shape]);
            end
            
            for i=1:length(index_array)
                j = index_array(i);
                x = self.get_data_(self.X, j);
                
                if (ndims(x) == 2)
                    batch_x(i, :) = x;
                elseif (ndims(x) == 3)
                    batch_x(i, :, :, :) = x;
                end
            
                if (~isempty(sync_gen))
                    xt = sync_gen.get_data_(sync_gen.X, j);
                    if (ndims(xt) == 2)
                        batch_xt(i, :) = xt;
                    elseif (ndims(xt) == 3)
                        batch_xt(i, :, :, :) = xt;
                    end
                end
            end
            
            % optionally save augmented images to disk for debugging purposes
            if (self.save_to_dir)
                % for i in range(current_batch_size):
                %     img = array_to_img(batch_x[i], self.data_format, scale=True)
                %     fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                %                                                       index=current_index + i,
                %                                                       hash=np.random.randint(1e4),
                %                                                       format=self.save_format)
                %     img.save(os.path.join(self.save_to_dir, fname))
                % end
            end
                
            % Perform the reshape operation on x if necessary
            if (self.input_vectorized)
                sz = size(batch_x);
                batch_x = reshape(batch_x, size(batch_x, 1), prod(sz(2:end)));
            end
            
            if (~isempty(sync_gen) && sync_gen.input_vectorized)
                % TODO: use np.ravel or x.ravel()
                sz = size(batch_xt);
                batch_xt = reshape(batch_xt, size(batch_xt, 1), prod(sz(2:end)));
            end
            
            % Process class_mode
            if (~ismember(self.class_mode, {'categorical', 'binary', 'sparse'}))
                if (~isempty(sync_gen))
                    data_batch = {batch_x, batch_xt};
                else    
                    data_batch = {batch_x, batch_x};
                end
                return;
            end
            
            assert(~isempty(self.y));
            classes = self.y(index_array);
            
            if (self.class_mode == 'sparse')
                batch_y = classes;
            elseif (self.class_mode == 'binary')
                batch_y = classes; %.astype('float32')
            elseif (self.class_mode == 'categorical')
                assert(~isempty(self.nb_class))
                batch_y = zeros([size(batch_x, 1) self.nb_class]);
                for i=1:length(classes)
                    label = classes(i);
                    batch_y(i, label) = 1.;
                end
            else
                if (~isempty(sync_gen))
                    data_batch = {batch_x, batch_xt};
                else    
                    data_batch = {batch_x, batch_x};
                end
                return;
            end               
            
            data_batch = {batch_x, batch_y};
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function release(self)
            % Release internal resources used by the iterator.
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function x = get_data_(self, X, j)
            % Load image from in memory database, pre-process and return.
            % 
            % Parameters
            % ----------
            % X : `array_like`
            %     Data matrix. Format Samples x ...
            % 
            % j : int
            %     Index of the data item to be featched.

            assert(numel(self.image_shape) == 3)
            x = self.X(j, :, :, :);
                    
            % x, ~ = self.image_data_generator.random_transform(x.astype('float32'));
            x = self.image_data_generator.standardize(x);            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

    






