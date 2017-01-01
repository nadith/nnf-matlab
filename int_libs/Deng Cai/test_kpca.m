function [accuracy] = test_kpca()

% Data_in = rand(10, 50);
% [Data_Out, eigvec] = kernelpca_tutorial(Data_in,5);
% 
% 
% Data_te = rand(10, 1);
% %% Using the Gaussian Kernel to construct the Kernel K
% % K(x,y) = -exp((x-y)^2/(sigma)^2)
% % K is a symmetric Kernel
% K = zeros(size(Data_in,2),1);
% for row = 1:size(Data_in,2)    
%     temp = sum(((Data_in(:,row) - Data_te).^2));
%     K(row,1) = exp(-temp); % sigma = 1    
% end
% 
% one_mat = ones(size(K));
% K_center = K - one_mat'*K - K*one_mat + one_mat*K*one_mat;
% clear K


options.KernelType = 'Gaussian';
options.t = 1;
options.ReducedDim = 4;
fea = rand(7,10);
[eigvector, eigvalue] = KPCA(fea, options);
feaTest = rand(3,10);
Ktest = constructKernel(feaTest, fea, options)
Y = Ktest*eigvector;


end