function [ img ] = im_pre_process( img, params)
%IM_PRE_PROCESS pre-process the image.
    % Imports
    import nnf.pp.*;

    if (~isfield(params, 'histeq')); params.histeq = false; end
    if (~isfield(params, 'normalize')); params.normalize = false; end
    if (~isfield(params, 'histmatch')); params.histmatch = false; end
    if (~isfield(params, 'cann_img')); params.cann_img = []; end

    if (params.histeq)
        img = hist_eq_ch(img, params.ch_axis);
    end

    % TODO: dtype changes
    % if (params.normalize)        
    %     img = bsxfun(@minus, double(img), mean(mean(img)));        
    % %         means = mean(mean(low_dim_img));
    % %         for i=1:numel(means)
    % %             low_dim_img(:, :, i) = low_dim_img(:, :, i) - means(i);
    % %         end
    % end
    
    % Histrogram matching
    if (params.histmatch && ~isempty(params.cann_img))
        img = hist_match_ch(img, params.cann_img, params.ch_axis);
    end        
end