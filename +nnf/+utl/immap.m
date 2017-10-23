function [image_map] = immap( X, varargin )
    %SHOW_IMAGE_MAP visualizes image data tensor in a grid.
    %
    % Parameters
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
    %
    % title : string, optional 
    %     figure title, (Default value = [])
    %
    %
    % Returns
    % -------
    % none : void
    %
    %
    % Examples
    % --------
    % Show an image grid of 5 rows and 8 cols (5x8 cells).
    % show_image_map(db, 5, 8)
    % show_image_map(db, 5, 'Cols', 8)
    % show_image_map(db, 'Rows', 5, 'Cols', 8)
    %
    % Show an image grid of 5 rows and 8 cols (5x8 cells). half resolution.
    % show_image_map(db, 5, 8, 0.5)    
    %
    % Show an image grid of 5 rows and 8 cols (5x8 cells). start from 10th image.
    % show_image_map(db, 5, 8, [], 10)

    % Copyright 2015-2016 Nadith Pathirage, Curtin University.
    
    % Set defaults
    rows = 1;
    cols = 1;
    scale = [];
    offset = 1;
    ws = struct;
    ftitle = [];

    % Handling varargs
    for i=1:numel(varargin)
        arg = varargin{i};
        if (isa('char', class(arg)))
            p = inputParser;
            addOptional(p, 'Rows', rows);
            addOptional(p, 'Cols', cols);
            addOptional(p, 'Scale', scale);
            addOptional(p, 'Offset', offset);
            addOptional(p, 'WS', ws);
            addOptional(p, 'Title', ftitle);
            parse(p, varargin{i:end});            
            rows = p.Results.Rows;
            cols = p.Results.Cols;
            scale = p.Results.Scale;
            offset = p.Results.Offset;
            ws = p.Results.WS;
            ftitle = p.Results.Title;
            break;            
        else
            % Read the params in order
            switch (i)
                case 1; rows = arg;
                case 2; cols = arg;
                case 3; scale = arg;
                case 4; offset = arg;
                case 5; ws = arg;
                case 6; ftitle = arg;
                otherwise
                    error('Suppplied parameters exceed maximum number of parameters allowed.')
            end
        end
    end

    % Set defaults for arguments
    if (~isfield(ws, 'height')); ws.height = 0; end
    if (~isfield(ws, 'width')); ws.width = 0; end
        
    % If `X` is a database with many images: 4D tensor in the format `H x W x CH x N`
    if (numel(size(X)) > 3) 
        [h, w, ch, n] = size(X);
        
    else % X: `H x W x CH`: In matlab last singular dimension is ignored by default.
        [h, w, ch] = size(X);
        n = 1;        
    end    
    if (~isfield(ws, 'color')) 
        if (ch > 1); ws.color = [0 0 0]; else; ws.color = 0; end
    end
    
    % Requested image count
    im_count = rows * cols;

    % Choose images with offset
    if (~isempty(scale) && ...
            ((isscalar(scale) && scale ~= 1) || (~isscalar(scale) && ~isequal(scale, [1 1]))))
        
        % Scale Operation, if scale is a scalar/vector
        if (isscalar(scale))
            newX = uint8(zeros(int32(h*scale), int32(w*scale), ch, im_count));
        else
            newX = uint8(zeros(int32(scale(1)), int32(scale(2)), ch, im_count));
        end
        
        % Set the end
        en = offset + im_count-1;
        if (en > n)
            en = n; 
            im_count = en - offset + 1;
        end
        
        for i = offset:en            
            % Resize the image (low dimension/resolution)            
            newX(:, :, :, i - (offset-1)) = imresize(X(:, :, :, i), scale);
        end
        
    else
        newX = uint8(zeros(h, w, ch, im_count));
        
        % Set the end
        en = offset + im_count-1;
        if (en > n)
            en = n; 
            im_count = en - offset + 1;            
        end   
                
        newX(:, :, :, 1:im_count) = X(:, :, :, offset:en);
    end
    
    % Building the grid
    [dim_y, dim_x, ~, ~] = size(newX);    
        
    % Whitespace information for building the grid
    ws_color = ws.color;
    
     % For RGB color build a color matrix (3D)
    if (~isscalar(ws.color))       
        ws_color = [];
        ws_color(:, :, 1) = ws.color(1);
        ws_color(:, :, 2) = ws.color(2);
        ws_color(:, :, 3) = ws.color(3);
    end
            
    GRID_H = (dim_y + ws.height) * rows - ws.height;
    GRID_W = (dim_x + ws.width) * cols - ws.width;
    image_map = uint8(ones(GRID_H, GRID_W, ch) .* ws_color);
       
    % Fill the grid
    for i = 1:rows
        for j = 1:cols
            im_index = (i-1) * cols + j;
            if (im_index > im_count); break; end;
            
            y_block = (i-1)*(dim_y + ws.height) + 1:(i-1)*(dim_y + ws.height) + dim_y;
            x_block = (j-1)*(dim_x + ws.width) + 1:(j-1)*(dim_x + ws.width) + dim_x;
            image_map(y_block, x_block, :) = newX(:, :, :, im_index);
        end        
        if (im_index > im_count); break; end
    end
        
    % Visualizing the grid
    imshow(image_map);
    
    % Figure title
    if (isempty(ftitle))
        title([int2str(dim_y) 'x' int2str(dim_x)]);
    else
        title(ftitle);
    end    
end
