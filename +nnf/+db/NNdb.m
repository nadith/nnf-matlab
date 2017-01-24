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
        matlab_db;      % Matlab Compatible db Format
        python_db;      % Python Compatible db Format
        features;
        im_ch_axis;     % Image channel axis
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
            %     Format of the database. (Default value = 1, start from 1).
            %             

            disp(['Costructor::NNdb ' name]);
            
            % Imports
            import nnf.db.Format; 
                        
            % Set defaults for arguments
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
        function init_db_fields(self)
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
        function data = get_data_at(self, si) 
            % GET_DATA_AT: gets data from database at i.
            %
            % Parameters
            % ----------
            % si : int
            %     Sample index.
            %             
            
            % Imports
            import nnf.db.Format;
            
            % Error handling for arguments
            assert(si <= self.n);            
                        
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
        function features = get_features(self, cls_lbl) 
            % 2D Feature Matrix (double)
            %
            % Parameters
            % ----------
            % cls_lbl : uint16, optional
            %     featres for class denoted by cls_lbl.
            %
            
            features = double(reshape(self.db, self.h * self.w * self.ch, self.n));
            
            % Select class
            if (nargin >= 2)
                 features = features(:, self.cls_lbl == cls_lbl);
            end
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
                    [~, ~, ~, n_per_class] = size(self.db);
                elseif (format == Format.H_N)
                    [~, n_per_class] = size(self.db);
                elseif (format == Format.N_H_W_CH)
                    [n_per_class, ~, ~, ~] = size(self.db);
                elseif (format == Format.N_H)
                    [n_per_class, ~] = size(self.db);
                end
                
            elseif (isempty(n_per_class))
                % Build n_per_class from cls_lbl
                [n_per_class, ~] = hist(cls_lbl,unique(cls_lbl))
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
            
            new_nndb = NNdb(name, self.db, self.n_per_class, self.build_cls_lbl, self.cls_lbl, self.format);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function show(self, cls_n, n_per_class, scale, offset) 
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
            % Examples
            % --------
            % Show first 5 subjects with 8 images per subject. (offset = 1)
            % .Show(5, 8)
            %
            % Show next 5 subjects with 8 images per subject, starting at (5*8 + 1)th image.
            % .Show(5, 8, [], 5*8 + 1)
            %
            
            % Imports
            import nnf.utl.immap;
            
            if (nargin >= 5)
                immap(self.db, cls_n, n_per_class, scale, offset);
            elseif (nargin >= 4)
                immap(self.db, cls_n, n_per_class, scale);
            elseif (nargin >= 3)
                immap(self.db, cls_n, n_per_class);
            elseif (nargin >= 2)
                immap(self.db, cls_n, 1);
            elseif (nargin >= 1)
                immap(self.db, 1, 1);
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
        function rgb2colors(self, normalized, to15, to22) 
            % RGB2COLORS: Convert RGB db to 15 or 22 color components.
            %
            
            % Set defaults for arguments
            if (nargin < 3), to15 = true; end
            if (nargin < 4), to22 = false; end
            
            % Error handling for arguments
            if (to15 && to22)
                warning('ARG_CONFLICT: to15, to22');
            end
            
            if (~to15 && ~to22)
                error('ARG_ERR:to15, to22: both are false');
            end
            
            %%% TRANFORMATION FUNCTIONS
            ColorTC = cell(1, 1);            
            ColorTC{1}=[1,0,0;0,1,0;0,0,1]; % RGB
            ColorTC{2}=[0.607,0.299,0.000;0.174,0.587,0.066;0.201,0.114,1.117]; %XYZ
            ColorTC{3}=[0.2900,0.5957,0.2115;0.5870,-0.2744,-0.5226;0.1140,-0.3213,0.3111]; %YIQ
            ColorTC{4}=[1/3,1/2,-1/2;1/3,0,1;1/3,-1/2,-1/2]; %III
            %YCbCr=[(0.2126*219)/255,(0.2126*224)/(1.8556*255),(0.5*224)/255;(0.7152*219)/255, ...
            %       (0.7152*224)/((1.8556*255)),-((0.7152*224)/(1.5748*255));..
            %       (0.0722*219)/255,(0.5*224)/255,-((0.0722*224)/(1.5748*255))];
            YCbCr_T   = (1/255) * [65.481 -37.797 112; 128.553 -74.203 -93.786; 24.966 112 -18.214];
            YCbCr_Off = (1/255) * [16 128 128];
            ColorTC{5}=[0.2990,-0.1471,0.6148;0.5870,-0.2888,-0.5148;0.1140,0.4359,-0.1000]; %YUV
            ColorTC{6}=[1,-1/3,-1/3;0,2/3,-1/3;0,-1/3,2/3]; %nRGB
            ColorTC{7}=[0.6070,-0.0343,-0.3940;0.1740,0.2537,-0.3280;0.2000,-0.2193,0.7220]; %nXYZ
                 
            % Build the transformation matrix
            transform = zeros(3, numel(ColorTC)*3);            
            for i=1:numel(ColorTC)                
                transform(:, 1+(i-1)*3:i*3) = ColorTC{i}; % transform=[RGB XYZ YIQ III YUV nRGB nXYZ];
            end
            
%             % not supported yet
%             for i=1:images
%                 
%                 YCBCR=rgb2ycbcr(img(:,:,:,i));
% 
%                 HSV=rgb2hsv(img(:,:,:,i));
%                 conv(:,:,1:3,i)=YCBCR(:,:,1:3);
%                 conv(:,:,4:6,i)=HSV(:,:,1:3);
% 
%             end 
%             conv1   = reshape(conv,row*col,6,images);

            fsize = self.h * self.w;
            rgb = double(reshape(self.db, fsize, self.ch, [])); 
            
            if (to22)
                % 3 + 3 for YCbCr, HSV
                tdb = zeros(fsize, size(transform, 2) + 3 + 3, images);
            else
                tdb = zeros(fsize, size(transform, 2), images);
            end           
            
            % Set Max, Min for normalization purpose
            maxT            = transform;
            maxT(maxT < 0)  = 0;
            channelMax      = ([255 255 255] * maxT);
            
            minT            = transform;
            minT(minT > 0)  = 0;
            channelMin      = ([255 255 255] * minT);
            
            % Required range
            newMax          = ones(1, size(transform, 2))*255;
            newMin          = ones(1, size(transform, 2))*0;
                        
            for i=1:self.n   
                temp = rgb(:,:,i)*transform;       
                
                if(normalized)           
                    %((x - channelMin) * ((newMax - newMin)/ (channelMax - channelMin))) + newMin
                    temp = bsxfun(@minus, temp, channelMin);
                    temp = bsxfun(@times, temp, (newMax - newMin)./ (channelMax - channelMin));
                    temp = bsxfun(@plus, temp, newMin);
                end          
                
                assert(uint8(max(max(temp))) <= max(newMax));
                assert(uint8(min(min(temp))) >= min(newMin));
                
                if (to22)
                    % YCbCr/HSV Transformation (done explicitely)
                    % Use this section only if the normalization
                    % range is [0, 255]
                    % temp2, temp3 will always be in the range [0, 255]
                    temp2 = reshape(rgb2ycbcr(reshape(rgb(:,:,i), self.h, self.w, [])), fsize, []);
                    %temp2 = rgb * YCbCr_T + repmat(YCbCr_Off, row*col, 1)                 
                    temp3 = reshape(rgb2hsv(reshape(rgb(:,:,i), self.h, self.w, [])), fsize, []);

                    assert(uint8(max(max(temp2))) <= max(newMax));
                    assert(uint8(min(min(temp2))) >= min(newMin));
                    assert(uint8(max(max(temp3))) <= max(newMax));
                    assert(uint8(min(min(temp3))) >= min(newMin));

                    tdb(:,:,i) = [temp temp2 temp3];
                    
                else
                    tdb(:,:,i) = uint8(temp);
                    
                end
            end
            clear rgb;
            
            % not supported yet
%             for i=1:images
%                 coo(:,22:27,i)=conv1(:,1:6,i);
%             end

            % Perform the selection (Mustapha's model)
            tdb2(:,1:6,:)   = tdb(:,1:6,:);
            tdb2(:,7:11,:)  = tdb(:,8:12,:);
            tdb2(:,12:13,:) = tdb(:,14:15,:);
            tdb2(:,14:15,:) = tdb(:,17:18,:);
            
            if (all22)
                tdb2(:,16:17,:) = tdb(:,20:21,:);
                tdb2(:,18:22,:) = tdb(:,23:27,:);
            end
            % % %  coo2(:,23:25,:)=dcs(:,:,:);            
            clear tdb;
            
            if (to22)
                self.db = reshape(tdb2, self.h, self.w, 22, self.n);
            else
                self.db = reshape(tdb2, self.h, self.w, 15, self.n);
            end        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end       
    
    methods 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Dependant property Implementations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.matlab_db(self) 
            % TODO: rollaxis/permute depending on the format
            
            value = self.n / self.n_per_class;
        end  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.python_db(self) 
            % TODO: rollaxis/permute depending on the format
            
            value = self.n / self.n_per_class;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.features(self) 
            % 2D Feature Matrix (double)
            import nnf.db.Format;
            
            % N x H x W x CH or N x H
            if (self.format == Format.N_H_W_CH || self.format == Format.N_H)
                value = double(reshape(self.db, self.n, self.h * self.w * self.ch));
            
            % H x W x CH x N or H x N
            elseif (self.format == Format.H_W_CH_N || self.format == Format.H_N)
                value = double(reshape(self.db, self.h * self.w * self.ch, self.n));

            else
                error('Unsupported db format')
            end
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


