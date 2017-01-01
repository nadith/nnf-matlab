function I = histmatch(T, S)
% T = Target
% S = Source 

%%
% For Grey Images
nT = size(T);
nS = size(S);
if (numel(nT) == 2 && numel(nS) == 2)
    H_T = imhist(T);
    I = histeq(S, H_T);
end

%%
% % For Colored Images    
% % Target
% Ihsv = rgb2hsv(T); 
% V_T = Ihsv(:,:,3);
% H_T = imhist(V_T);
% 
% % Source
% Ihsv = rgb2hsv(S); 
% V_S = Ihsv(:,:,3);
% H_S = imhist(V_T);
% Ihsv(:,:,3) = histeq(V_S, H_T);
% H_I = imhist(Ihsv(:,:,3));
% I = hsv2rgb(Ihsv);
% 
% if isa(S, 'uint8')
%     I = uint8(255 * I);
% elseif isa(S, 'uint16')
%     I = uint16(65535 * I);
% end



%%
% Debuging
% x_y_range = [0 260 0 600];
% 
% figure;
% subplot(2,3,1), imshow(T);
% subplot(2,3,4), plot(H_T);
% axis(x_y_range);
% 
% subplot(2,3,2), imshow(S);
% subplot(2,3,5), plot(H_S);
% axis(x_y_range);
% 
% subplot(2,3,3), imshow(I);
% subplot(2,3,6), plot(H_I);
% axis(x_y_range);

end