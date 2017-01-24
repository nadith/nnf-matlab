function [cimg] = hist_match_ch(cimg, cann_cimg, ch_axis)
%Histogram matching on individual color channel.
    nch = size(cimg, ch_axis);
    for ich=[1:nch]
        if (ch_axis == 1)
            cimg(ich, :, :) = imhistmatch(cimg(ich, :, :), cann_cimg(ich, :, :));
        elseif (ch_axis == 2)
            cimg(:, ich, :) = imhistmatch(cimg(:, ich, :), cann_cimg(:, ich, :));
        elseif  (ch_axis == 3)
            cimg(:, :, ich) = imhistmatch(cimg(:, :, ich), cann_cimg(:, :, ich));
        end
    end
end