classdef ImageDataGenerator < handle
    % Generate minibatches with
    % real-time data augmentation.
    % 
    % % Arguments
    %     featurewise_center: set input mean to 0 over the dataset.
    %     samplewise_center: set each sample mean to 0.
    %     featurewise_std_normalization: divide inputs by std of the dataset.
    %     samplewise_std_normalization: divide each input by its std.
    %     zca_whitening: apply ZCA whitening.
    %     rotation_range: degrees (0 to 180).
    %     width_shift_range: fraction of total width.
    %     height_shift_range: fraction of total height.
    %     shear_range: shear intensity (shear angle in radians).
    %     zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
    %         in the range [1-z, 1+z]. A sequence of two can be passed instead
    %         to select this range.
    %     channel_shift_range: shift range for each channels.
    %     fill_mode: points outside the boundaries are filled according to the
    %         given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
    %         is 'nearest'.
    %     cval: value used for points outside the boundaries when fill_mode is
    %         'constant'. Default is 0.
    %     horizontal_flip: whether to randomly flip images horizontally.
    %     vertical_flip: whether to randomly flip images vertically.
    %     rescale: rescaling factor. If None or 0, no rescaling is applied,
    %         otherwise we multiply the data by the value provided
    %         (before applying any other transformation).
    %     preprocessing_function: function that will be implied on each input.
    %         The function will run before any other modification on it.
    %         The function should take one argument: one image (Numpy tensor with rank 3),
    %         and should output a Numpy tensor with the same shape.
    %     data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
    %         (the depth) is at index 1, in 'channels_last' mode it is at index 3.
    %         It defaults to the `image_data_format` value found in your
    %         Keras config file at `~/.keras/keras.json`.
    %         If you never set it, then it will be "channels_last".
    %
   
    properties
        featurewise_center;
        samplewise_center;
        featurewise_std_normalization;
        samplewise_std_normalization;
        zca_whitening;
        rotation_range;
        width_shift_range;
        height_shift_range;
        shear_range;
        zoom_range;
        channel_shift_range;
        fill_mode;
        cval;
        horizontal_flip;
        vertical_flip;

        augment;
        rounds;
        seed;
    end

    properties (SetAccess = private)        
        rescale;
        preprocessing_function;
                
        data_format;
        channel_index;
        row_index;
        col_index;
    end
        
    properties (SetAccess = protected)
        mean;
        std;
        principal_components;
        map_min_max;
        whiten;
    end
        
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = ImageDataGenerator(params)
            
            % Imports
            import nnf.core.K;
            
            if (isempty(params))
                self.featurewise_center = false;
                self.samplewise_center = false;
                self.featurewise_std_normalization = false;
                self.samplewise_std_normalization = false;
                self.zca_whitening = false;
                self.rotation_range = 0.;
                self.width_shift_range = 0.;
                self.height_shift_range = 0.;
                self.shear_range = 0.;
                zoom_range = 0.;
                self.channel_shift_range = 0.;
                self.fill_mode = 'nearest';
                self.cval = 0.;
                self.horizontal_flip = false;
                self.vertical_flip = false;
                rescale = [];
                preprocessing_function = [];
                data_format = [];

                % To invoke fit() function to calculate 
                % featurewise_center, featurewise_std_normalization and zca_whitening
                self.augment = false;
                self.rounds = 1;
                self.seed = [];

            else
                if (params.isKey('featurewise_center')); self.featurewise_center = params.get('featurewise_center'); else; self.featurewise_center = false; end
                if (params.isKey('samplewise_center')); self.samplewise_center = params.get('samplewise_center'); else; self.samplewise_center = false; end
                if (params.isKey('featurewise_std_normalization')); self.featurewise_std_normalization = params.get('featurewise_std_normalization'); else; self.featurewise_std_normalization = false; end
                if (params.isKey('zca_whitening')); self.zca_whitening = params.get('zca_whitening'); else; self.zca_whitening = false; end
                if (params.isKey('rotation_range')); self.rotation_range = params.get('rotation_range'); else; self.rotation_range = 0.; end
                if (params.isKey('width_shift_range')); self.width_shift_range = params.get('width_shift_range'); else; self.width_shift_range = 0.; end
                if (params.isKey('height_shift_range')); self.height_shift_range = params.get('height_shift_range'); else; self.height_shift_range = 0.; end
                if (params.isKey('shear_range')); self.shear_range = params.get('shear_range'); else; self.shear_range = 0.; end
                if (params.isKey('zoom_range')); zoom_range = params.get('zoom_range'); else; zoom_range = 0.; end
                if (params.isKey('channel_shift_range')); self.channel_shift_range = params.get('channel_shift_range'); else; self.channel_shift_range = 0.; end
                if (params.isKey('fill_mode')); self.fill_mode = params.get('fill_mode'); else; self.fill_mode = 'nearest'; end
                if (params.isKey('cval')); self.cval = params.get('cval'); else; self.cval = 0.; end
                if (params.isKey('horizontal_flip')); self.horizontal_flip = params.get('horizontal_flip'); else; self.horizontal_flip = false; end
                if (params.isKey('vertical_flip')); self.vertical_flip = params.get('vertical_flip'); else; self.vertical_flip = false; end
                if (params.isKey('rescale')); rescale = params.get('rescale'); else; rescale = []; end
                if (params.isKey('preprocessing_function')); preprocessing_function = params.get('preprocessing_function'); else; preprocessing_function = []; end
                if (params.isKey('data_format')); data_format = params.get('data_format'); else; data_format = []; end

                % To invoke fit() function to calculate 
                % featurewise_center, featurewise_std_normalization and zca_whitening
                if (params.isKey('augment')); self.augment = params.get('augment'); else; self.augment = false; end
                if (params.isKey('rounds')); self.rounds = params.get('rounds'); else; self.rounds = 1; end
                if (params.isKey('seed')); self.seed = params.get('seed'); else; self.seed = []; end
            end

            if (isempty(data_format))
                data_format = K.image_data_format;
            end
        
            % self.__dict__.update(locals())
            self.mean = [];
            self.std = [];
            self.principal_components = [];
            self.rescale = rescale;
            self.preprocessing_function = preprocessing_function;
            
            if (~ismember(data_format, {'channels_last', 'channels_first'}))
                error(['data_format should be "channels_last" (channel after row and ' ...
                                 'column) or "channels_first" (channel before row and column). ' ...
                                 'Received arg: ' data_format]);
            end
            
            self.data_format = data_format;
            if strcmp(data_format, 'channels_first')
                self.channel_index = 2;
                self.row_index = 3;
                self.col_index = 4;
            end
            
            if strcmp(data_format, 'channels_last')
                self.channel_index = 4;
                self.row_index = 2;
                self.col_index = 3;
            end
            
            if isscalar(zoom_range)
                self.zoom_range = [1 - zoom_range, 1 + zoom_range];
            elseif length(zoom_range) == 2
                self.zoom_range = [zoom_range(0), zoom_range(1)];
            else
                error(['zoom_range should be a float or ' ...
                                 'a tuple or list of two floats. ' ...
                                 'Received arg: ' num2str(zoom_range)]);
            end
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function x = standardize(self, x)
            if self.preprocessing_function
                x = self.preprocessing_function(x);
            end            
            if self.rescale
                x = double(x) * self.rescale;
            end
            
            % x is a single image, so it doesn't have image number at index 0
            img_channel_index = self.channel_index - 1;
            if self.samplewise_center
                M = mean(x, img_channel_index);
                x = bsxfun(@minus, x, M);
            end
            
            if self.samplewise_std_normalization
                S = std(x, img_channel_index);
                x = bsxfun(@rdivide, x, S); % + K.epsilon()
            end
            
            % if self.featurewise_center:
            %     if self.mean is not None:
            %         x -= self.mean
            %     else:
            %         warnings.warn('This ImageDataGenerator specifies '
            %                       '`featurewise_center`, but it hasn\'t'
            %                       'been fit on any training data. Fit it '
            %                       'first by calling `.fit(numpy_data)`.')
            % if self.featurewise_std_normalization:
            %     if self.std is not None:
            %         x /= (self.std + 1e-7)
            %     else:
            %         warnings.warn('This ImageDataGenerator specifies '
            %                       '`featurewise_std_normalization`, but it hasn\'t'
            %                       'been fit on any training data. Fit it '
            %                       'first by calling `.fit(numpy_data)`.')
            % if self.zca_whitening:
            %     if self.principal_components is not None:
            %         flatx = np.reshape(x, (x.size))
            %         whitex = np.dot(flatx, self.principal_components)
            %         x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            %     else:
            %         warnings.warn('This ImageDataGenerator specifies '
            %                       '`zca_whitening`, but it hasn\'t'
            %                       'been fit on any training data. Fit it '
            %                       'first by calling `.fit(numpy_data)`.')
            % return x
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [x, xt] = random_transform(self, x, xt)            
            
            % Set default values
            if (nargin < 3); xt = []; end
            
            
            % # x is a single image, so it doesn't have image number at index 0
            % img_row_index = self.row_index - 1
            % img_col_index = self.col_index - 1
            % img_channel_index = self.channel_index - 1
            % 
            % # use composition of homographies to generate final transform that needs to be applied
            % if self.rotation_range:
            %     theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            % else:
            %     theta = 0
            % rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
            %                             [np.sin(theta), np.cos(theta), 0],
            %                             [0, 0, 1]])
            % if self.height_shift_range:
            %     tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
            % else:
            %     tx = 0
            % 
            % if self.width_shift_range:
            %     ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
            % else:
            %     ty = 0
            % 
            % translation_matrix = np.array([[1, 0, tx],
            %                                [0, 1, ty],
            %                                [0, 0, 1]])
            % if self.shear_range:
            %     shear = np.random.uniform(-self.shear_range, self.shear_range)
            % else:
            %     shear = 0
            % shear_matrix = np.array([[1, -np.sin(shear), 0],
            %                          [0, np.cos(shear), 0],
            %                          [0, 0, 1]])
            % 
            % if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            %     zx, zy = 1, 1
            % else:
            %     zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
            % zoom_matrix = np.array([[zx, 0, 0],
            %                         [0, zy, 0],
            %                         [0, 0, 1]])
            % 
            % transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
            % 
            % h, w = x.shape[img_row_index], x.shape[img_col_index]
            % 
            % if (xt is not None):
            %     assert(h==xt.shape[img_row_index] and w==xt.shape[img_col_index])
            % 
            % transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            % x = apply_transform(x, transform_matrix, img_channel_index,
            %                     fill_mode=self.fill_mode, cval=self.cval)
            % 
            % xt = None if (xt is None) else apply_transform(xt, transform_matrix, img_channel_index,
            %                                                 fill_mode=self.fill_mode, cval=self.cval)
            % 
            % if self.channel_shift_range != 0:
            %     x = random_channel_shift(x, self.channel_shift_range, img_channel_index)
            %     xt = None if (xt is None) else random_channel_shift(xt, self.channel_shift_range, img_channel_index)
            % 
            % if self.horizontal_flip:
            %     if np.random.random() < 0.5:
            %         x = flip_axis(x, img_col_index)
            %         xt = None if (xt is None) else flip_axis(xt, img_col_index)
            % 
            % if self.vertical_flip:
            %     if np.random.random() < 0.5:
            %         x = flip_axis(x, img_row_index)
            %         xt = None if (xt is None) else flip_axis(xt, img_row_index)
            % 
            % # TODO:
            % # channel-wise normalization
            % # barrel/fisheye
            % return x, xt
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function fit(self, X, augment, rounds, seed)
            % Required for featurewise_center, featurewise_std_normalization
            % and zca_whitening.

            % # Arguments
            %     X: Numpy array, the data to fit on. Should have rank 4.
            %         In case of grayscale data,
            %         the channels axis should have value 1, and in case
            %         of RGB data, it should have value 3.
            %     augment: Whether to fit on randomly augmented samples
            %     rounds: If `augment`,
            %         how many augmentation passes to do over the data
            %     seed: random seed.
            % 
            
            % Set default values
            if (nargin < 5); seed = []; end
            if (nargin < 4); rounds = 1; end
            if (nargin < 3); augment = false; end
            
            % X = np.asarray(X)
            % if X.ndim != 4:
            %     raise ValueError('Input to `.fit()` should have rank 4. '
            %                      'Got array with shape: ' + str(X.shape))
            % if X.shape[self.channel_index] not in {1, 3, 4}:
            %     raise ValueError(
            %         'Expected input to be images (as Numpy array) '
            %         'following the dimension ordering convention "' + self.data_format + '" '
            %         '(channels on axis ' + str(self.channel_index) + '), i.e. expected '
            %         'either 1, 3 or 4 channels on axis ' + str(self.channel_index) + '. '
            %         'However, it was passed an array with shape ' + str(X.shape) +
            %         ' (' + str(X.shape[self.channel_index]) + ' channels).')
            % 
            
            
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

            if (~isempty(seed))
                np.random.seed(seed);
            end
            
            % X = np.copy(X)
            % if augment:
            %     aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            %     for r in range(rounds):
            %         for i in range(X.shape[0]):
            %             aX[i + r * X.shape[0]] = self.random_transform(X[i])
            %     X = aX
            
            if self.featurewise_center
                
                % mean across all samples, row_index, and col_index
                if (strcmp(self.data_format, 'channel_last'))
                    M = mean(X, 1); % Image no. axis
                    M = mean(M, 2); % self.row_index
                    self.mean = mean(M, 3); % self.col_index
                    
                else % if ('channel_first')
                    M = mean(X, 1); % Image no. axis
                    M = mean(M, 3); % self.row_index
                    self.mean = mean(M, 4); % self.col_index
                end
                
                X = bsxfun(@minus, X, self.mean);
            end
            
            if self.featurewise_std_normalization
                
                % std across all samples, row_index, and col_index
                if (strcmp(self.data_format, 'channel_last'))
                    S = std(X, 1); % Image no. axis
                    S = std(S, 2); % self.row_index
                    self.mean = std(S, 3); % self.col_index
                    
                else % if ('channel_first')
                    S = std(X, 1); % Image no. axis
                    S = std(S, 3); % self.row_index
                    self.std = mean(S, 4); % self.col_index
                end
                
                X = bsxfun(@rdivide, X, self.std);
                % X /= (self.std + K.epsilon())
            end
            
            % if self.zca_whitening
            %     flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            %     sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            %     U, S, V = linalg.svd(sigma)
            %     self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end


    
