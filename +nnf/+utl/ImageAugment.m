classdef ImageAugment < handle
    % ImageAugment represents util class to perform image data augmentation.

	% Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    methods (Access = public, Static)        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndb_aug = gauss_data_gen(nndb, info, merge)
            % GAUSS_DATA_GEN: Augment the dataset with centered gaussian at class mean or each image in the class.
            % 
            % Parameters
            % ----------
            % nndb : :obj:`NNdb`
            %     In memory database to iterate.
            % 
            % info : :obj:`dict`, optional
            %     Provide additional information to perform guass data generation.
            %     (Default value = {}).
            %
            %     Info Params (Defaults)
            %     ----------------------
            %     - inf.samples_per_class = 2     # Samples per class required
            %     - inf.center_cls_mean = False   # Use class mean image centered gaussian
            %     - inf.noise_ratio = 0.2         # Ratio from class std (ratio * class_std)
            %     - inf.std = None                # Explicit defitnition of std ('noise_ratio' is ignored)
            %
            % merge : bool
            %     Merge the augmented database to `nndb` object.
            %

            % Imports
            import nnf.db.NNdb;

            % Set defaults for arguments
            if (nargin < 3); merge = false; end
            if (nargin < 2); info = struct; end        
            if (~isfield(info, 'center_cls_mean')); info.center_cls_mean = false; end
            if (~isfield(info, 'samples_per_class')); info.samples_per_class = 5; end
            if (~isfield(info, 'noise_ratio')); info.noise_ratio = 0.2; end
            if (~isfield(info, 'std')); info.std = []; end % Force standard deviation
            % to avoid the calculation of the std over the samples in the class.
            % 'noise_ratio' will be ignored

            % Initialize necessary variables
            use_cls_mean = info.center_cls_mean;
            max_samples_per_class = info.samples_per_class;
            noise_ratio = info.noise_ratio;
            force_std = info.std;
            fsize = nndb.h* nndb.w* nndb.ch;

            % Create a new database to save the generated data
            nndb_aug = NNdb('augmented', [], [], false, [], nndb.format);

            for i=1:nndb.cls_n

                % Fetch samples per class i
                n_per_class = nndb.n_per_class(i);

                st = nndb.cls_st(i);
                en = st + uint32(n_per_class) - 1;

                % Fetch input database for class i    
                I = nndb.features(:, st:en);

                % By default, gauss distributions are centered at each image
                M = double(I);

                % Gauss distributions are centered at the mean of each class
                if (use_cls_mean)
                    M = mean(double(I), 2); 
                end    

                % Initialize a 2D matrix to store augmented data for class
                I_aug = cast(zeros(fsize, ceil(max_samples_per_class/n_per_class)*n_per_class), class(nndb.db));

                % Initialize the index variable for above matrix
                j = 1;

                % Keep track of current sample count per class
                cur_samples_per_class = 0;
                while (cur_samples_per_class < max_samples_per_class)            
                    r = randn(fsize, n_per_class); % r is from gaussian (m=0, std=1)

                    if (isempty(force_std)) % Caculate the std from the samples in the class i
                        I_aug(:, 1+(j-1)*n_per_class: j*n_per_class) = cast(bsxfun(@plus, M, noise_ratio * (std(double(I), 0, 2) .* r)), class(nndb.db));      
                    else
                         I_aug(:, 1+(j-1)*n_per_class: j*n_per_class) = cast(bsxfun(@plus, M, force_std .* r), class(nndb.db));      
                    end

                    cur_samples_per_class = cur_samples_per_class + n_per_class;
                    j = j + 1;
                end

                I_aug = nndb_aug.features_to_data(I_aug(:, 1:max_samples_per_class), nndb.h, nndb.w, nndb.ch, class(nndb.db));
                nndb_aug.add_data(I_aug);
                nndb_aug.update_attr(true, max_samples_per_class);
            end

            if (merge)
                nndb_aug = nndb.merge(nndb_aug);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
