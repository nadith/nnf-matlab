function [ img ] = slinda_pprocess( img )
%SLINDA_PPROCESS Summary of this function goes here
[h, w, ~] = size(img);
img(:, :, 1) = uint8(zeros(h, w));
end

