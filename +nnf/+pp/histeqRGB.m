function J = histeqRGB(I)
Ihsv = rgb2hsv(I); % convert from RGB space to HSV space

% V is roughly equivalent to intensity, bear in mind that

% V is now of type double, with values in the range [0-1] V = Ihsv(:,:,3);
V = Ihsv(:,:,3);

% perform histogram equalization on the V channel,

% overwriting the original data Ihsv(:,:,3) = histeq(V);
Ihsv(:,:,3) = histeq(V);

J = hsv2rgb(Ihsv);

if isa(I, 'uint8')
    J = uint8(255 * J);
elseif isa(I, 'uint16')
    J = uint16(65535 * J);
end