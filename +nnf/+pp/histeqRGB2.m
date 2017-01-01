function J = histeqRGB2(I)

ycbcrmap = rgb2ycbcr(I);


% V is roughly equivalent to intensity, bear in mind that

% V is now of type double, with values in the range [0-1] V = Ihsv(:,:,3);
V = ycbcrmap(:,:,1);

% perform histogram equalization on the V channel,

% overwriting the original data Ihsv(:,:,3) = histeq(V);
ycbcrmap(:,:,1) = histeq(V);

J = ycbcr2rgb(ycbcrmap);
end