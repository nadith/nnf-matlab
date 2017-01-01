function [image_map] = immap( X, rows, cols, scale, offset )
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
    %
    % Show an image grid of 5 rows and 8 cols (5x8 cells). half resolution.
    % show_image_map(db, 5, 8, 0.5)
    %
    % Show an image grid of 5 rows and 8 cols (5x8 cells). start from 10th image.
    % show_image_map(db, 5, 8, [], 10)

    % Copyright 2015-2016 Nadith Pathirage, Curtin University.

    if (numel(size(X)) ~= 4)
        error('ARG_ERR: X: 4D tensor in the format H x W x CH x N');
    end
    if (nargin < 2)
        error('ARG_ERR: rows, cols: undefined');
    end
    if (nargin < 3)
        error('ARG_ERR: cols: undefined');
    end
    
    % Set defaults for arguments
    if (nargin < 4), scale = []; end
    if (nargin < 5), offset = 1; end
    
    % Fetch no of color channels
    [h, w, ch, n] = size(X);

    % Requested image count
    im_count = rows * cols;

    % Choose images with offset
    if (~isempty(scale) && scale ~= 1)        
        % Scale Operation        
        newX = uint8(zeros(h*scale, w*scale, ch, im_count));  
        
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
    image_map = uint8(zeros(dim_y * rows, dim_x * cols, ch));

    % Fill the grid
    for i = 1:rows
        for j = 1:cols
            im_index = (i-1) * cols + j;
            if (im_index > im_count); break; end;
            image_map((i-1)*dim_y + 1:(i)*dim_y, (j-1)*dim_x + 1:(j)*dim_x, :) = newX(:, :, :, im_index);
        end        
        if (im_index > im_count); break; end;
    end
        
    % Visualizing the grid
    imshow(image_map);
    
    % Figure Title
    title([int2str(dim_y) 'x' int2str(dim_x)]);
    
end
