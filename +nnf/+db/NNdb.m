classdef NNdb < handle
    % NNDB represents database for NNFramwork.
    % IMPL_NOTES: Pass by reference class.
    %
    % i.e
    % Database with same no. of images per class, with build class idx
    % nndb = NNdb('any_name', imdb, 8, true)
    %
    % Database with varying no. of images per class, with build class idx
    % nndb = NNdb('any_name', imdb, [4 3 3 1], true)
    %
    % Database with given class labels
    % nndb = NNdb('any_name', imdb, [4 3], false, [1 1 1 1 2 2 2])
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
	properties (SetAccess = public)
        name;           % (s) Name of nndb object
        db;             % (M) Actual Database   
        format;         % (s) Current Format of The Database
        
        h;              % (s) Height (Y dimension)
        w;              % (s) Width (X dimension)
        ch;             % (s) Channel Count
        n;              % (s) Sample Count
        
        n_per_class;    % (v) No of images per class (classes may have different no. of images)  
        cls_st;         % (v) Class Start Index  (internal use, can be used publicly)
        build_cls_lbl   % (s) Build the class labels or not.
        cls_lbl;        % (v) Class Index Array
        cls_n;          % (s) Class Count
  	end
                
    properties (SetAccess = public, Dependent)
        db_convo_th;    % db compatible for convolutional networks.
        db_convo_tf;    % db compatible for convolutional networks.        
        db_scipy;       % db compatible for scipy library.
        features_scipy; % 2D feature matrix (double) compatible for scipy library.
        db_matlab;      % db compatible for matlab.
        features;       % 2D feature matrix (double) compatible for matlab.
        zero_to_one;    % nndb converted to 0-1 range.
        im_ch_axis;     % Image channel index for an image.
    end
    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function nndb = load_from_dir(dirpath, db_name) 
            % LOAD_FROM_DIR: Load images from a directory. 
            % 
            % Parameters
            % ----------
            % dirpath : string
            %     Path to directory of images sorted in folder per each class.
            % 
            % db_name : string, optional
            %     Name for the nndb object returned. (Default value = 'DB').
            %

            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;

            % Set defaults for arguments
            if (nargin < 2); db_name = 'DB'; end

            % Init empty NNdb to collect images
            nndb = NNdb(db_name, [], [], false, [], Format.H_W_CH_N);

            cls_structs = dir(dirpath);
            cls_structs = cls_structs(~ismember({cls_structs.name},{'.','..'})); % exclude '.' and '..'

            % Sort the folder names (class names)
            [~,ndx] = natsortfiles({cls_structs.name}); % indices of correct order
            cls_structs = cls_structs(ndx);             % sort structure using indices

            % Iterator
            for cls_i = 1:length(cls_structs)

                cls_name = cls_structs(cls_i).name;
                cls_dir = fullfile(dirpath, cls_name);

                % img_structs = dir (fullfile(ims_dir, '*.jpg')); % Only jpg files
                img_structs = dir(cls_dir);
                img_structs = img_structs(~ismember({img_structs.name},{'.','..'})); % exclude '.' and '..'

                % Sort the image files (file names)
                [~,ndx] = natsortfiles({img_structs.name}); % indices of correct order
                img_structs = img_structs(ndx);             % sort structure using indices            

                is_new_class = true;
                for cls_img_i = 1 : length(img_structs)
                    img_name = img_structs(cls_img_i).name;
                    img = imread(fullfile(cls_dir, img_name));

                    % Update NNdb
                    nndb.add_data(img);
                    nndb.update_attr(is_new_class);                        
                    is_new_class = false;                       
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        function self = NNdb(name, db, n_per_class, build_cls_lbl, cls_lbl, format) 
            % Constructs a nndb object.
            %
            % Parameters
            % ----------
            % db : 4D tensor -uint8
            %     Data tensor that contains images.
            % 
            % n_per_class : vector -uint16 or scalar, optional
            %     No. images per each class. (Default value = []).
            % 
            % build_cls_lbl : bool, optional
            %     Build the class indices or not. (Default value = false).
            % 
            % cls_lbl : vector -uint16 or scalar, optional
            %     Class index array. (Default value = []).
            % 
            % format : nnf.db.Format, optinal
            %     Format of the database. (Default value = Format.H_W_CH_N, refer `nnf.db.Format`).
            %             
            
            % Imports
            import nnf.db.Format;
            import nnf.utl.disp;
            
            self.name = name;
            disp(['Costructor::NNdb ' name]);
            
            % Set defaults for arguments
            if (nargin < 2), db = []; end
            if (nargin < 3), n_per_class = []; end
            if (nargin < 4), build_cls_lbl = false; end
            if (nargin < 5), cls_lbl = []; end
            if (nargin < 6), format = Format.H_W_CH_N; end     
            
            % Error handling for arguments
            if (isscalar(cls_lbl))
                error('ARG_ERR: cls_lbl: vector indicating class for each sample');
            end
            
            if (isempty(db))
                self.db = []; 
                self.n_per_class = [];
                self.build_cls_lbl = build_cls_lbl;
                self.cls_lbl = cls_lbl;
                self.format = format;
                self.h = 0; self.w = 1; self.ch = 1; self.n = 0;                
                self.cls_st = [];                
                self.cls_n = 0;
                return
            end
                        
            % Set values for instance variables
            self.set_db(db, n_per_class, build_cls_lbl, cls_lbl, format);
        end
      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function nndb = merge(self, nndb) 
            % MERGE: `nndb` instance with `self` instance.
            % 
            % Parameters
            % ----------
            % nndb : :obj:`NNdb`
            %     NNdb object that represents the dataset.
            %
        
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
            
            assert(self.h == nndb.h && self.w == nndb.w && self.ch == nndb.ch);
            assert(self.cls_n == nndb.cls_n);
            assert(strcmp(class(self.db), class(nndb.db)))
            assert(self.format == nndb.format)
        
            if (self.format == Format.H_W_CH_N)
                db = cast(zeros(self.h, self.w, self.ch, self.n + nndb.n), class(self.db));
            elseif (self.format == Format.H_N)
                db = cast(zeros(self.h * self.w * self.ch, self.n + nndb.n), class(self.db));
            elseif (self.format == Format.N_H_W_CH)
                db = cast(zeros(self.n + nndb.n, self.h, self.w, self.ch), class(self.db));
            elseif (self.format == Format.N_H)
                db = cast(zeros(self.n + nndb.n, self.h * self.w * self.ch), class(self.db));
            end
            
            cls_lbl = uint16(zeros(1, self.n + nndb.n));
            en = 0;
            for i=1:self.cls_n
                % Fetch data from db1
                cls_st = self.cls_st(i);
                cls_end = cls_st + uint32(self.n_per_class(i)) - uint32(1);
                
                st = en + 1;
                en = st + self.n_per_class(i) - 1;
                cls_lbl(st:en) = i .* uint16(ones(1, self.n_per_class(i)));
                
                if (self.format == Format.H_W_CH_N)
                    db(:, :, :, st:en) = self.db(:, :, :, cls_st:cls_end);
                elseif (self.format == Format.H_N)
                    db(:, st:en) = self.db(:, cls_st:cls_end);
                elseif (self.format == Format.N_H_W_CH)
                    db(st:en, :, :, :) = self.db(cls_st:cls_end, :, :, :);
                elseif (self.format == Format.N_H)
                    db(st:en, :) = self.db(cls_st:cls_end, :);
                end            
                
                % Fetch data from db2
                cls_st = nndb.cls_st(i);
                cls_end = cls_st + uint32(nndb.n_per_class(i)) - uint32(1);
                
                st = en + 1;
                en = st + nndb.n_per_class(i) - 1;
                cls_lbl(st:en) = i .* uint16(ones(1, nndb.n_per_class(i)));
                
                if (self.format == Format.H_W_CH_N)
                    db(:, :, :, st:en) = nndb.db(:, :, :, cls_st:cls_end);
                elseif (self.format == Format.H_N)
                    db(:, st:en) = nndb.db(:, cls_st:cls_end);
                elseif (self.format == Format.N_H_W_CH)
                    db(st:en, :, :, :) = nndb.db(cls_st:cls_end, :, :, :);
                elseif (self.format == Format.N_H)
                    db(st:en, :) = nndb.db(cls_st:cls_end, :);
                end
            end
                        
            nndb = NNdb('merged', db, [], false, cls_lbl, self.format);            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndb = concat_features(self, nndb) 
            % AUGMENT_FEATURES: augment `nndb` instance with `self` instance.
            %   Both `nndb` and self instances must be in the format Format.H_N or Format.N_H (2D databases)
            % 
            % Parameters
            % ----------
            % nndb : :obj:`NNdb`
            %     NNdb object that represents the dataset.
            %
        
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
            
            assert(self.n == nndb.n);
            assert(isequal(self.n_per_class, nndb.n_per_class));
            assert(self.cls_n == nndb.cls_n);
            assert(strcmp(class(self.db), class(nndb.db)))
            assert(self.format == nndb.format)
            assert(self.format == Format.H_N || self.format == Format.N_H);        
                        
            if (self.format == Format.H_N)
                db = [self.db; nndb.db];
            
            elseif (self.format == Format.N_H)                
                db = [self.db nndb.db];
            end
            
            nndb = NNdb('features_augmented', db, self.n_per_class, true, [], self.format);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = convert_format(self, format, h, w, ch) 
            % CONVERT_FORMAT: Convert the format of this `nndb` object to target format.
            %   h, w, ch are conditionally optional, used only when converting 2D nndb to 4D nndb
            %   formats.            
            %
            % Parameters
            % ----------
            % format : nnf.db.Format
            %     Target format of the database.
            % 
            % h : int, optional under conditions
            %     height to be used when converting from Format.H_N to Format.H_W_CH_N.
            % 
            % w : int, optional under conditions
            %     width to be used when converting from Format.H_N to Format.H_W_CH_N.
            % 
            % ch : int, optional under conditions
            %     channels to be used when converting from Format.H_N to Format.H_W_CH_N.
            %
            
            % Imports
            import nnf.db.Format;
            
            % Fetch datatype
            dtype = class(self.db);
            
            if (self.format == Format.H_W_CH_N || self.format == Format.N_H_W_CH)
                if (format == Format.H_N)
                    self.db = cast(self.features, dtype);
                    self.h = size(self.db, 1);
                    self.w = 1;
                    self.ch = 1;
                    
                elseif (format == Format.N_H)
                    self.db = cast(self.features', dtype);
                    self.h = size(self.db, 2);
                    self.w = 1;
                    self.ch = 1;
                    
                elseif (format == Format.H_W_CH_N)
                   self.db = self.db_matlab;                
                
                elseif (format == Format.N_H_W_CH)
                    self.db = self.db_scipy;
                end
                
                self.format = format;
                
            elseif (self.format == Format.H_N || self.format == Format.N_H)                
                
                if (format == Format.H_W_CH_N)
                    self.db = reshape(self.db_matlab, h, w, ch, self.n);
                    self.h = h;
                    self.w = w;
                    self.ch = ch;
                
                elseif (format == Format.N_H_W_CH)
                    self.db = reshape(self.db_matlab, self.n, h, w, ch);
                    self.h = h;
                    self.w = w;
                    self.ch = ch;
                    
                elseif (format == Format.H_N)
                    self.db = self.db_matlab;
                    
                elseif (format == Format.N_H)
                    self.db = self.db_scipy;
                    
                end
                
                self.format = format;
            end
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function update_attr(self, is_new_class, sample_n) 
            % UPDATE_NNDB_ATTR: update self attributes. Used when building the nndb dynamically.
            % 
            % Can invoke this method for every item added (default sample_n=1) or 
            % batch of items added for a given class (sample_n > 1).
            % 
            % Parameters
            % ----------
            % is_new_class : bool
            %     Recently added data item/batch belongs to a new class.
            % 
            % sample_n : int
            %     Recently added data batch size. (Default value = 1).
            % 
            % Examples
            % --------
            % Using this method to update the attibutes of nndb dynamically.
            %
            % >> nndb = NNdb("EMPTY_NNDB", [], [], false, [], format=Format.H_W_CH_N)
            % >> data = rand(30, 30, 1, 100)   # data tensor for each class
            % >> nndb.add_data(data)
            % >> nndb.update_attr(true, 100)
            %
                
            % Set defaults
            if (nargin < 3); sample_n = 1; end
            
            % Initialize db related fields
            self.init_db_fields__();
                
            % Set class start, and class counts of nndb
            if (is_new_class)                
                % Set class start(s) of nndb, dynamic expansion
                cls_st = self.n - sample_n + 1; % start of the most recent item addition
                if (isempty(self.cls_st)); self.cls_st = uint32([]); end
                self.cls_st = cat(2, self.cls_st, uint32([cls_st]));

                % Set class count
                self.cls_n = self.cls_n + 1;

                % Set n_per_class(s) of nndb, dynamic expansion
                n_per_class = 0;
                if (isempty(self.n_per_class)); self.n_per_class = uint16([]); end
                self.n_per_class = cat(2, self.n_per_class, uint16([n_per_class]));
            end    

            % Increment the n_per_class current class
            self.n_per_class(end) = self.n_per_class(end) + sample_n;

            % Set class label of nndb, dynamic expansion
            cls_lbl = self.cls_n;
            if (isempty(self.cls_lbl)); self.cls_lbl = uint16([]); end
            self.cls_lbl = cat(2, self.cls_lbl, uint16([cls_lbl]));
        end       
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data = get_data_at(self, si) 
            % GET_DATA_AT: gets data from database at i.
            %
            % Parameters
            % ----------
            % si : int
            %     Sample index or range.
            %             
            
            % Imports
            import nnf.db.Format;
            
            % Error handling for arguments
            assert(isempty(find(si > self.n)));
                        
            % Get data according to the format
            if (self.format == Format.H_W_CH_N)
                data = self.db(:, :, :, si);
            elseif (self.format == Format.H_N)
                data = self.db(:, si);
            elseif (self.format == Format.N_H_W_CH)
                data = self.db(si, :, :, :);
            elseif (self.format == Format.N_H)
                data = self.db(si, :);
            end
        end
        
       	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function add_data(self, data) 
            % ADD_DATA: adds data into the database.
            %
            % Parameters
            % ----------
            % data : `array_like`
            %     Data to be added.
            %
            % Notes
            % -----
            % Dynamic allocation for the data tensor.
            % 
            
            % Imports
            import nnf.db.Format;

            % Add data according to the format (dynamic allocation)
            if (self.format == Format.H_W_CH_N)
                if (isempty(self.db))
                    self.db = data;
                else
                    self.db = cat(4, self.db, data);
                end

            elseif (self.format == Format.H_N)
                if (isempty(self.db))
                    self.db = data;
                else
                    self.db = cat(2, self.db, data);
                end               

            elseif (self.format == Format.N_H_W_CH)
                if (isempty(self.db))
                    self.db = data;
                else
                    self.db = cat(1, self.db, data);
                end 

            elseif (self.format == Format.N_H)
                if (isempty(self.db))
                    self.db = data;
                else
                    self.db = cat(1, self.db, data);
                end 
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data = features_to_data(self, features, h, w, ch, dtype) 
            % FEATURES_TO_DATA: converts the feature matrix to `self` compatible data format and type.
            %   h, w, ch, dtype are conditionally optional, used only when self.db = []. 
            %
            % Parameters
            % ----------
            % features : `array_like`
            %     2D feature matrix (double) compatible for matlab. (F_SIZE x SAMPLES)
            % 
            % h : int, optional under conditions
            %     height to be used when self.db = [].  
            % 
            % w : int, optional under conditions
            %     width to be used when self.db = [].   
            % 
            % ch : int, optional under conditions
            %     no of channels to be used when self.db = [].   
            % 
            % dtype : int, optional under conditions
            %     data type to be used when self.db = [].
            %           
                  
            % Set default for arguments
            if (nargin < 6); dtype = []; end
            if (nargin < 5); ch = []; end
            if (nargin < 4); w = []; end
            if (nargin < 3); h = []; end
            
            assert((isempty(self.db) && ~(isempty(h) || isempty(w) || isempty(ch) || isempty(dtype))) ||...
                (~isempty(self.db)));

            if (~isempty(self.db))
                h = self.h;
                w = self.w;
                ch = self.ch;
                dtype = class(self.db);
            end            
            
            % Imports
            import nnf.db.Format;
            
            % Sample count
            n = size(features, 2);

            % Add data according to the format (dynamic allocation)
            if (self.format == Format.H_W_CH_N)
                data = reshape(features, h, w, ch, n);

            % elseif (self.format == Format.H_N)
            %     data = data;

            elseif (self.format == Format.N_H_W_CH)
                data = reshape(features, h, w, ch, n);
                data = permute(data, [4 1 2 3]);

            elseif (self.format == Format.N_H)
                data = data';                
            end
            
            data = cast(data, dtype);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function features = get_features(self, cls_lbl, norm) 
            % Get normalized 2D Feature matrix (double) for specified class labels.
            %
            % Parameters
            % ----------
            % cls_lbl : uint16, optional
            %     featres for class denoted by cls_lbl.
            %
            % norm : string, optional
            %     'l1', 'l2', 'max', normlization for each column. (Default value = []).
            %
            
            features = self.features;
            
            % Set default for arguments
            if (nargin < 2); cls_lbl = []; end
            if (nargin < 3); norm = []; end
            
            % Select class
            if (~isempty(cls_lbl))
                 features = features(:, self.cls_lbl == cls_lbl);
            end
            
            if (~isempty(norm))
                assert(strcmp(norm, 'l2')); %TODO: implement for other norms
                features = normc(features);
            end
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [features, m] = get_features_mean_diff(self, cls_lbl, m) 
            % Get the 2D feature mean difference matrix for specified class labels and mean.
            % 
            % Parameters
            % ----------
            % cls_lbl : scalar, optional
            %       Class index array. (Default value = None).
            % 
            % m : `array_like`, optional
            %       Mean vector to calculate feature mean difference. (Default value = None).
            % 
            % Returns
            % -------
            % features : `array_like` -double
            %       2D feature difference matrix.
            %
            % m : `array_like` -double
            %       Calculated mean.
            %            
            
            if (nargin < 2); cls_lbl = []; end  
            if (nargin < 3); m = []; end  
            
            features = self.get_features(cls_lbl)
            if (isempty(m)); m = mean(features, 2); end
           
            features = (features - repmat(m, [1, self.n]));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        function set_db(self, db, n_per_class, build_cls_lbl, cls_lbl, format) 
            % SET_DB: sets database and update relevant instance variables.
            % i.e (db, format, cls_lbl, cls_n, etc)
            %
            % Parameters
            % ----------
            % db : 4D tensor -uint8
            %     Data tensor that contains images.
            % 
            % n_per_class : vector -uint16 or scalar, optional
            %     No. images per each class. (Default value = []).
            %     If (n_per_class=[] and cls_lbl=[]) then n_per_class = total image count
            % 
            % build_cls_lbl : bool, optional
            %     Build the class indices or not. (Default value = false).
            % 
            % cls_lbl : vector -uint16 or scalar, optional
            %     Class index array. (Default value = []).
            % 
            % format : nnf.db.Format, optinal
            %     Format of the database. (Default value = 1, start from 1).
            %
            
            % Imports
            import nnf.db.Format; 

            % Error handling for arguments
            if (isempty(db))
                error('ARG_ERR: n_per_class: undefined');
            end                      
            if (isempty(format))
                error('ARG_ERR: format: undefined');
            end            
            if (~isempty(cls_lbl) && build_cls_lbl)
                warning('ARG_CONFLICT: cls_lbl, build_cls_lbl');
            end
            
            % Set defaults for n_per_class
            if (isempty(n_per_class) && isempty(cls_lbl))
                if (format == Format.H_W_CH_N)
                    [~, ~, ~, n_per_class] = size(db);
                elseif (format == Format.H_N)
                    [~, n_per_class] = size(db);
                elseif (format == Format.N_H_W_CH)
                    [n_per_class, ~, ~, ~] = size(db);
                elseif (format == Format.N_H)
                    [n_per_class, ~] = size(db);
                end
                
            elseif (isempty(n_per_class))
                % Build n_per_class from cls_lbl
                [n_per_class, ~] = hist(cls_lbl,unique(double(cls_lbl)));
            end
            
        	% Set defaults for instance variables
            self.db = []; self.format = [];
            self.h = 0; self.w = 1; self.ch = 1; self.n = 0;
            self.n_per_class = [];
            self.cls_st = [];
            self.cls_lbl = [];
            self.cls_n = 0;
            
            % Set values for instance variables
            self.db     = db;
            self.format = format;
            self.build_cls_lbl = build_cls_lbl;
            
            % Set h, w, ch, np according to the format    
            if (format == Format.H_W_CH_N)
                [self.h, self.w, self.ch, self.n] = size(self.db);
            elseif (format == Format.H_N)
                [self.h, self.n] = size(self.db);
            elseif (format == Format.N_H_W_CH)
                [ self.n, self.h, self.w, self.ch] = size(self.db);
            elseif (format == Format.N_H)
                [self.n, self.h] = size(self.db);
            end
                   
            % Set class count, n_per_class, class start index
            if (isscalar(n_per_class))
                if (mod(self.n, n_per_class) > 0)
                    error('Total image count (n) is not divisable by image per class (n_per_class)')
                end            
                self.cls_n = self.n / n_per_class;
                self.n_per_class =  uint16(repmat(n_per_class, 1, self.cls_n));
                tmp = uint32(self.n_per_class .* uint16(1:self.cls_n) + uint16(ones(1, self.cls_n)));
                self.cls_st = [1 tmp(1:end-1)];
            else
                self.cls_n = numel(n_per_class);
                self.n_per_class =  uint16(n_per_class);
                
                if (self.cls_n > 0)
                    self.cls_st = uint32(zeros(1, numel(n_per_class)));
                    self.cls_st(1) = 1;
                    
                    if (self.cls_n > 1)                    
                        st = n_per_class(1) + 1;
                        for i=2:self.cls_n
                            self.cls_st(i) = st;
                            st = st + n_per_class(i);                    
                        end
                    end
                end  
            end
            
            % Set class labels
            self.cls_lbl = cls_lbl;  
                  
            % Build uniform cls labels if cls_lbl is not given  
            if (build_cls_lbl && isempty(cls_lbl))
                    self.build_sorted_cls_lbl();
            end             
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function build_sorted_cls_lbl(self) 
            % BUILD_sorted_CLS_LBL: Builds a sorted class indicies/labels  for samples
            
            n_per_class = self.n_per_class;            
                        
            % Each image should belong to a class            
            cls_lbl = uint16(zeros(1, self.n));    
            st = 1;
            for i = 1:self.cls_n
                cls_lbl(st: st + n_per_class(i) - 1) = uint16(ones(1, n_per_class(i)) * i);                
                st = st + n_per_class(i);
            end
            
            % Set values for instance variables
            self.cls_lbl = cls_lbl;           
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function new_nndb = clone(self, name) 
            % CLONE: Creates a copy of this NNdb object
            %
            % Imports 
            import nnf.db.NNdb;
            
            new_nndb = NNdb(name, self.db, self.n_per_class, self.build_cls_lbl, self.cls_lbl, self.format);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function show_ws(self, varargin) 
            % SHOW: Visualizes the db in a image grid with whitespacing.
            %
            % Parameters
            % ----------
            % cls_n : int, optional
            %     No. of classes.
            % 
            % n_per_class : int, optional
            %     Images per class.
            % 
            % scale : int, optional
            %     Scaling factor. (Default value = [])
            % 
            % offset : int, optional
            %     Image index offset to the dataset. (Default value = 1)
            %
            % ws : struct, optional
            %     whitespace between images in the grid.
            %
            %     Whitespace Structure (with defaults)
            %     -----------------------------------
            %     ws.height = 5;                    % whitespace in height, y direction (0 = no whitespace)  
            %     ws.width  = 5;                    % whitespace in width, x direction (0 = no whitespace)  
            %     ws.color  = 0 or 255 or [R G B];  % (255 = white)
            %
            % title : string, optional 
            %     figure title, (Default value = [])
            %
            % Examples
            % --------
            % Show first 5 subjects with 8 images per subject. (offset = 1)
            % nndb.show_ws(5, 8)
            %
            % Show next 5 subjects with 8 images per subject, starting at (5*8 + 1)th image.
            % nndb.show_ws(5, 8, 'Offset', 5*8 + 1)
            %
            
            % Imports
            import nnf.utl.immap;
            
            ws = [];        
            for i=1:numel(varargin)
                arg = varargin{i};
                if (isa('char', class(arg)))
                    for j=i:2:numel(varargin)
                        arg = varargin{j};
                        assert(isa('char', class(arg)));
                        if (strcmp(arg, 'WS'))
                            ws = varargin{j+1};
                            if (~isfield(ws, 'height')); ws.height = 5; end
                            if (~isfield(ws, 'width')); ws.width = 5; end
                            if (~isfield(ws, 'color')) 
                                if (self.ch > 1); ws.color = [255 0 0]; else; ws.color = 255; end
                            end
                            varargin{j+1} = ws;
                            break;
                        end
                    end
                    break;            
                else
                    % Read the params in order
                    if (i == 5) % 5th parameter denotes the ws structure
                        ws = arg;
                        if (~isfield(ws, 'height')); ws.height = 5; end
                        if (~isfield(ws, 'width')); ws.width = 5; end
                        if (~isfield(ws, 'color')) 
                            if (self.ch > 1); ws.color = [255 0 0]; else; ws.color = 255; end
                        end
                        arg = ws; 
                    end
                    varargin{i} = arg;
                end
            end

            if (isempty(ws))
                ws = struct;
                ws.height = 5;
                ws.width = 5;
                if (self.ch > 1); ws.color = [255 0 0]; else; ws.color = 255; end

                varargin{end + 1} = 'WS';
                varargin{end + 1} = ws;
            end

            immap(self.db_matlab, varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function show(self, varargin)
            % SHOW: Visualizes the db in a image grid.
            %
            % Parameters
            % ----------
            % cls_n : int, optional
            %     No. of classes.
            % 
            % n_per_class : int, optional
            %     Images per class.
            % 
            % scale : int, optional
            %     Scaling factor. (Default value = [])
            % 
            % offset : int, optional
            %     Image index offset to the dataset. (Default value = 1)
            %
            % ws : struct, optional
            %     whitespace between images in the grid.
            %
            %     Whitespace Structure (with defaults)
            %     -----------------------------------
            %     ws.height = 0;                    % whitespace in height, y direction (0 = no whitespace)  
            %     ws.width  = 0;                    % whitespace in width, x direction (0 = no whitespace)  
            %     ws.color  = 0 or 255 or [R G B];  % (0 = black)
            %
            % title : string, optional 
            %     figure title, (Default value = [])
            %
            % Examples
            % --------
            % Show first 5 subjects with 8 images per subject. (offset = 1)
            % nndb.show(5, 8)
            %
            % Show next 5 subjects with 8 images per subject, starting at (5*8 + 1)th image.
            % nndb.show(5, 8, 'Offset', 5*8 + 1)
            %
            
            % Imports
            import nnf.utl.immap;           
            immap(self.db_matlab, varargin{:});      
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function save(self, filepath) 
            % Save images to a matfile. 
            % 
            % Parameters
            % ----------
            % filepath : string
            %     Path to the file.
            %
            
            imdb_obj.db = self.db_matlab;
            imdb_obj.class = self.cls_lbl;
            imdb_obj.im_per_class = self.n_per_class;
            save(filepath, 'imdb_obj', '-v7.3');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function save_compressed(self, filepath) 
            % Save images to a matfile. 
            % 
            % Parameters
            % ----------
            % filepath : string
            %     Path to the file.
            %
            
            imdb_obj.db = self.db_matlab;
            
            unq_n_per_class = unique(self.n_per_class);
            if isscalar(unq_n_per_class)
                imdb_obj.im_per_class = unq_n_per_class;
                imdb_obj.class = [];
            else            
                imdb_obj.im_per_class = self.n_per_class;
                imdb_obj.class = self.cls_lbl;
            end
            
            save(filepath, 'imdb_obj', '-v7.3');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function save_to_dir(self, filepath, create_cls_dir) 
            % Save images to a directory. 
            % 
            % Parameters
            % ----------
            % path : string
            %     Path to directory.
            % 
            % create_cls_dir : bool, optional
            %     Create directories for individual classes. (Default value = True).
            %
            
            % Set defaults
            if (nargin < 3); create_cls_dir = true; end
            
            % Make a new directory to save images
            if (~isempty(filepath) && exist(filepath, 'dir') == 0)
                mkdir(filepath);
            end
            
            img_i = 1;
            for cls_i=1:self.cls_n

                cls_name = num2str(cls_i); 
                if (create_cls_dir && exist(fullfile(filepath, cls_name), 'dir') == 0)
                    mkdir(fullfile(filepath, cls_name));
                end

                for cls_img_i=1:self.n_per_class(cls_i)
                    if (create_cls_dir)
                        img_name = num2str(cls_img_i);
                        imwrite(self.get_data_at(img_i), fullfile(filepath, cls_name, [img_name '.jpg']), 'jpg');
                    else                
                        img_name = [cls_name '_' num2str(cls_img_i)];
                        imwrite(self.get_data_at(img_i), fullfile(filepath, [img_name '.jpg']), 'jpg');
                    end
                    
                    img_i = img_i + 1;
                end
            end
        end
            
       	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function plot(self, n, offset) 
            % PLOT: plots the features.
            %   2D and 3D plots are currently supported.
            %
            % Parameters
            % ----------
            % n : int, optional
            %     No. of samples to visualize. (Default value = self.n)
            % 
            % offset : int, optional
            %     Sample index offset. (Default value = 1)
            %
            %
            % Examples
            % --------
            % plot 5 samples. (offset = 1)
            % .plot(5, 8)
            %
            % plot 5 samples starting from 10th sample
            % .plot(5, 10)
            %                        
            
            % Set defaults for arguments
            if (nargin < 2), n = self.n; end
            if (nargin < 3), offset = 1; end
            
            X = self.features;
            fsize = size(X, 1);
            
            % Error handling
            if (fsize > 3)
                error(['self.h = ' num2str(self.h) ', must be 2 for (2D) or 3 for (3D) plots']);
            end
            
            % Draw with colors if labels are avaiable
            if (~isempty(self.cls_lbl))
                for i=1:self.cls_n
                    
                    % Set st and en for class i
                    st = self.cls_st(i);
                    en = st + uint32(self.n_per_class(i)) - 1;
                    
                    % Break
                    if (st > offset + n - 1); break; end
                    
                    % Draw samples starting at offset
                    if (en > offset)
                        st = offset; 
                    else
                        continue;
                    end
                    
                    % Draw only n samples
                    if (en > offset + n - 1); en = offset + n - 1; end
                    
                    % Draw 2D or 3D plot
                    if (fsize == 2)
                        c = self.cls_lbl(st:en);
                        s = scatter(X(1, st:en), X(2, st:en), 25, c, 'filled', 'MarkerEdgeColor', 'k');
                        s.LineWidth = 0.1;
                        
                    elseif (fsize == 3)
                        c = self.cls_lbl(st:en);
                        s = scatter3(X(1, st:en), X(2, st:en), X(3, st:en), 25, c, ...
                                                            'filled', 'MarkerEdgeColor', 'k');
                        s.LineWidth = 0.1;                        
                    end
                end
                
                hold off;
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function init_db_fields__(self) 
            import nnf.db.Format;
            
            if (self.format == Format.H_W_CH_N)
                [self.h, self.w, self.ch, self.n] = size(self.db);

            elseif (self.format == Format.H_N)
                self.w = 1;
                self.ch = 1;
                [self.h, self.n] = size(self.db);

            elseif (self.format == Format.N_H_W_CH)
                [self.n, self.h, self.w, self.ch] = size(self.db);

            elseif (self.format == Format.N_H)
                self.w = 1;
                self.ch = 1;
                [self.n, self.h] = size(self.db);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Dependant property Implementations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function db = get.db_convo_th(self)
            % db compatible for convolutional networks.

            % Imports
            import nnf.db.Format; 
            
            % N x CH x H x W
            if (self.format == Format.N_H_W_CH || self.format == Format.H_W_CH_N)
                db = permute(self.db_scipy, [1 4 2 3]);
                
            % N x 1 x H x 1
            elseif (self.format == Format.N_H || self.format == Format.H_N)
                db = reshape(self.db_scipy, self.n, 1, self.h, 1);

            else
                error('Unsupported db format');
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function db = get.db_convo_tf(self)
            % db compatible for convolutional networks.
            
            % Imports
            import nnf.db.Format; 
            
            % N x H x W x CH
            if (self.format == Format.N_H_W_CH || self.format == Format.H_W_CH_N)
                db = self.db_scipy;

            % N x H
            elseif (self.format == Format.N_H || self.format == Format.H_N)
                db = self.db_scipy(:, :, 1, 1);

            else
                error('Unsupported db format');
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function db = get.db_scipy(self)
            % db compatible for scipy library.

            % Imports
            import nnf.db.Format; 
            
            % N x H x W x CH or N x H  
            if (self.format == Format.N_H_W_CH || self.format == Format.N_H)
                db = self.db;
                
            % H x W x CH x N
            elseif (self.format == Format.H_W_CH_N)
                db = permute(self.db,[4 1 2 3]);                

            % H x N
            elseif (self.format == Format.H_N)
                db = permute(self.db,[2 1]);
                
            else
                error('Unsupported db format');
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function db = get.features_scipy(self)
            % 2D feature matrix (double) compatible for scipy library.   
            db = double(reshape(self.db_scipy, self.n, self.h * self.w * self.ch));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function db = get.db_matlab(self)
            % db compatible for matlab.
            
            % Imports
            import nnf.db.Format; 

            % H x W x CH x N or H x N  
            if (self.format == Format.H_W_CH_N || self.format == Format.H_N)
                db = self.db;

            % N x H x W x CH
            elseif (self.format == Format.N_H_W_CH)
                db = permute(self.db,[2 3 4 1]);

            % N x H
            elseif (self.format == Format.N_H)
                db = permute(self.db,[2 1]);

            else
                raise Exception("Unsupported db format");
            end
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function db = get.features(self) 
            % 2D feature matrix (double) compatible for matlab.
            db = double(reshape(self.db_matlab, self.h * self.w * self.ch, self.n));
        end  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function nndb = get.zero_to_one(self) 
            % db converted to 0-1 range. database data type will be converted to double.
            
            % Imports
            import nnf.db.NNdb;
            
            % Construct a new object
            nndb = NNdb([self.name ' (0-1)'], double(self.db)/255, self.n_per_class, false, self.cls_lbl, self.format);            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.im_ch_axis(self) 
            % Get image channel index for an image.
            % 
            % Exclude the sample axis.
            %
               
            % Imports
            import nnf.db.Format;
            
            if (self.format == Format.H_W_CH_N)
                value = 3;
            elseif (self.format == Format.H_N)
                value = 0;
            elseif (self.format == Format.N_H_W_CH)
                value = 3;
            elseif (self.format == Format.N_H)
                value = 0;
            else
                error('Unsupported db format')
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
       
end


