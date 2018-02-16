function [nndb] = rgb2colors(nndb, normalized, to15, to22) 
    % RGB2COLORS: Convert RGB db to 15 or 22 color components.
    % Ref: 
       
    %
    % TODO: Parameters
    % ----------
    % X : array_like -uint8
    %     2D Data tensor that contains images.
    % 
    %     Format for color/grey images: (H x W x CH x N).
    % 
    % rows : int
    %     Number of rows in the grid.
    % 
    % cols : int
    %     Number of columns in the grid.
    % 
    % scale : int, optional
    %     Scale factor. (Default value = [], no resize operation required).
    % 
    % offset : int, optional
    %     Offset to the first image (Default value = 1, start from 1).
    % 
    % ws : struct, optional
    %     whitespace between images in the grid.
    %
    %     Whitespace Structure (with defaults)
    %     -----------------------------------
    %     ws.height = 0;                    % whitespace in height, y direction (0 = no whitespace)  
    %     ws.width  = 0;                    % whitespace in width, x direction (0 = no whitespace)  
    %     ws.color  = 0 or 255 or [R G B];  % (0 = black)
    
    
    % Imports
    import nnf.db.NNdb;
    import nnf.db.Format;

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

    % % not supported yet
    % for i=1:images
    % 
    %     YCBCR=rgb2ycbcr(img(:,:,:,i));
    % 
    %     HSV=rgb2hsv(img(:,:,:,i));
    %     conv(:,:,1:3,i)=YCBCR(:,:,1:3);
    %     conv(:,:,4:6,i)=HSV(:,:,1:3);
    % 
    % end 
    % conv1   = reshape(conv,row*col,6,images);

    fsize = nndb.h * nndb.w;
    rgb = double(reshape(nndb.db, fsize, nndb.ch, [])); 

    if (to22)
        % 3 + 3, To store YCbCr, HSV color spaces respectively
        tdb = zeros(fsize, size(transform, 2) + 3 + 3, nndb.n);
    else
        tdb = zeros(fsize, size(transform, 2), nndb.n);
    end           

    % Set Max, Min for normalization (each channel) purpose
    maxT            = transform;
    maxT(maxT < 0)  = 0;
    channel_max      = ([255 255 255] * maxT);

    minT            = transform;
    minT(minT > 0)  = 0;
    channel_min      = ([255 255 255] * minT);

    % Required range
    new_max          = ones(1, size(transform, 2))*255;
    new_min          = ones(1, size(transform, 2))*0;

    for i=1:nndb.n
        img = rgb(:,:,i);
        temp = img * transform;       

        % Normalize each channel with respect to their max and min values => 0 - 255
        if(normalized)           
            %((x - channelMin) * ((newMax - newMin)/ (channelMax - channelMin))) + newMin
            temp = bsxfun(@minus, temp, channel_min);
            temp = bsxfun(@times, temp, (new_max - new_min)./ (channel_max - channel_min));
            temp = bsxfun(@plus, temp, new_min);
        end          

        assert(uint8(max(max(temp))) <= max(new_max));
        assert(uint8(min(min(temp))) >= min(new_min));

        if (to22)
            % YCbCr/HSV Transformation (done explicitely)
            % Use this section only if the normalization
            % range is [0, 255]
            % temp_ycbcr, temp_hsv will always be in the range [0, 255]
            temp_ycbcr = reshape(rgb2ycbcr(reshape(img, nndb.h, nndb.w, [])), fsize, []);
            %temp2 = rgb * YCbCr_T + repmat(YCbCr_Off, row*col, 1)                 
            temp_hsv = reshape(rgb2hsv(reshape(img, nndb.h, nndb.w, [])), fsize, []);

            assert(uint8(max(max(temp_ycbcr))) <= max(new_max));
            assert(uint8(min(min(temp_ycbcr))) >= min(new_min));
            assert(uint8(max(max(temp_hsv))) <= max(new_max));
            assert(uint8(min(min(temp_hsv))) >= min(new_min));

            tdb(:,:,i) = uint8([temp temp_ycbcr temp_hsv]);

        else
            tdb(:,:,i) = uint8(temp);

        end
    end
    clear rgb;

    % not supported yet
    % for i=1:images
    %     coo(:,22:27,i)=conv1(:,1:6,i);
    % end

    % Perform the selection (Mustapha's model)
    tdb2(:,1:6,:)   = tdb(:,1:6,:);
    tdb2(:,7:11,:)  = tdb(:,8:12,:);
    tdb2(:,12:13,:) = tdb(:,14:15,:);
    tdb2(:,14:15,:) = tdb(:,17:18,:);

    if (to22)
        tdb2(:,16:17,:) = tdb(:,20:21,:);
        tdb2(:,18:22,:) = tdb(:,23:27,:);
    end
    % % %  coo2(:,23:25,:)=dcs(:,:,:);            
    clear tdb;
    
    h = nndb.h;
    w = nndb.w;
    n = nndb.n;
    if (to22)
        ch = 22;
    else
        ch = 15;
    end
    
    % Add data according to the format (dynamic allocation)
    if (nndb.format == Format.H_W_CH_N)
        data = reshape(tdb2, h, w, ch, n);

    elseif (self.format == Format.H_N)
        data = reshape(tdb2, h * w * ch, n);

    elseif (nndb.format == Format.N_H_W_CH)
        data = reshape(tdb2, h, w, ch, n);
        data = permute(data, [4 1 2 3]);

    elseif (nndb.format == Format.N_H)
        data = reshape(tdb2, h * w * ch, n);
        data = data';                
    end
    
    nndb = NNdb([nndb.name '- MultiColor'], data, nndb.n_per_class, false, nndb.cls_lbl, nndb.format);            
end
        