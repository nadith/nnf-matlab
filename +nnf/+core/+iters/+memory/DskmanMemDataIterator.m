classdef DskmanMemDataIterator < nnf.core.iters.DskmanDataIterator
    % DskmanMemDataIterator represents the diskman iterator for in memory databases.
    % 
    % Attributes
    % ----------
    % nndb : :obj:`NNdb`
    %     Database to iterate.
    % 
    % save_to_dir_ : str
    %     Path to directory of processed data.
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties (SetAccess = public)
        nndb;        
    end
    
    properties (SetAccess = protected)
        save_to_dir_;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = DskmanMemDataIterator(pp_params)
            % Construct a DskmanMemDataIterator instance.
            % 
            % Parameters
            % ----------
            % pp_params : :obj:`dict`
            %     Pre-processing parameters for :obj:`ImageDataPreProcessor`.
            %
            self = self@nnf.core.iters.DskmanDataIterator(pp_params);
            self.nndb = [];
            self.save_to_dir_ = [];
            
            % INHERITED: Whether to read the data
            % self.read_data_
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init_params(self, nndb, save_dir)
            % Initialize parameters for :obj:`DskmanMemDataIterator` instance.
            % 
            % Parameters
            % ----------
            % nndb : :obj:`NNdb`
            %     Database to iterate.
            % 
            % save_dir : str, optional
            %     Path to directory of processed data. (Default value = None)
            %
            
            % Set default values
            if (nargin < 3); save_dir = []; end
            
            self.nndb = nndb;
            self.save_to_dir_ = save_dir;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function clone(self)
        % Create a copy of this object.
            assert(false) % Currently not implemented
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function im_ch_axis = get_im_ch_axis(self)
            % Image channel axis.
            im_ch_axis = self.nndb.im_ch_axis;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function release_(self)
            % Release internal resources used by the iterator.
            release_@nnf.core.iters.DskmanDataIterator();
            self.nndb = [];
            self.save_to_dir_ = [];
        end        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [cimg, frecord] = get_cimg_frecord_in_next_(self, cls_idx, col_idx)
            % Get image and file record (frecord) at cls_idx, col_idx.
            % 
            % Parameters
            % ----------
            % cls_idx : int
            %     Class index. Belongs to `union_cls_range`.
            % 
            % col_idx : int
            %     Column index. Belongs to `union_col_range`.
            % 
            % Returns
            % -------
            % `array_like`
            %     Color image.
            % 
            % :obj:`list`
            %     file record. [file_path, file_position, class_label]
        	%
            if (cls_idx > self.nndb.cls_n)
            	error(['Class:' num2str(cls_idx) ' is missing in the database.']);
            end

            if (col_idx > self.nndb.n_per_class(cls_idx))
            	error(['Class:' num2str(cls_idx) ' ImageIdx:' num2str(col_idx)...
                            ' is missing in the database.']);
            end
            
            % Calulate the image index for the database
            im_idx = self.nndb.cls_st(cls_idx) + uint32(col_idx) - 1;
            
            cimg = [];
            if (self.read_data_)
                cimg = self.nndb.get_data_at(im_idx);
            end
            
            fpath_to_save = [];
            if (~isempty(self.save_to_dir_))
                fpath_to_save = fullfile(self.save_to_dir_, ['image_' num2str(im_idx) '.jpg']);
            end
            
            frecord = {fpath_to_save, [], uint16(cls_idx)}; % [fpath, fpos, cls_lbl]
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function valid = is_valid_cls_idx_(self, cls_idx, show_warning)
            % Check the validity of class index.
            % 
            % Parameters
            % ----------
            % cls_idx : int
            %     Class index. Belongs to `union_cls_range`.
            % 
            % Returns
            % -------
            % bool
            %     True if valid. False otherwise.
            %
            if (nargin < 3)
                show_warning = true;
            end
            
            if (cls_idx > self.nndb.cls_n && show_warning)
                warning(['Class:' num2str(cls_idx) ' is missing in the database']);
            end
            
            valid = cls_idx <= self.nndb.cls_n;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function valid = is_valid_col_idx_(self, cls_idx, col_idx, show_warning)
            % Check the validity of column index of the class denoted by cls_idx.
            % 
            % Parameters
            % ----------
            % cls_idx : int
            %     Class index. Belongs to `union_cls_range`.
            % 
            % col_idx : int
            %     Column index. Belongs to `union_col_range`.
            % 
            % Returns
            % -------
            % bool
            %     True if valid. False otherwise.
            %
            assert(cls_idx <= self.nndb.cls_n);
            
            if (nargin < 3)
                show_warning = true;
            end
            
            if (col_idx > self.nndb.n_per_class(cls_idx) && show_warning)
                warning(['Class:' num2str(cls_idx) ' ImageIdx:' num2str(col_idx) 
                        ' is missing in the database']);
            end

            valid = col_idx <= self.nndb.n_per_class(cls_idx);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function n_per_class = get_n_per_class_(self, cls_idx)
            % Get no of images per class.
            % 
            % Parameters
            % ----------
            % cls_idx : int
            %     Class index. Belongs to `union_cls_range`.
            % 
            % Returns
            % -------
            % int
            %     no of samples per class.
            %
            assert(cls_idx <= self.nndb.cls_n);
            n_per_class = self.nndb.n_per_class(cls_idx);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
