function [cimg] = hist_eq_ch(cimg, ch_axis)
%Histogram equalization on individual color channel.
    nch = size(cimg, ch_axis);
    for ich=[1:nch]
        if (ch_axis == 1)
            cimg(ich, :, :) = histeq(cimg(ich, :, :));
        elseif (ch_axis == 2)
            cimg(:, ich, :) = histeq(cimg(:, ich, :));
        elseif  (ch_axis == 3)
            cimg(:, :, ich) = histeq(cimg(:, :, ich));
        end
    end
end