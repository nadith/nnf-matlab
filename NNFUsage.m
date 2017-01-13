
import nnf.db.NNdb;
import nnf.db.DbSlice;
import nnf.db.NNPatch;

nndb = NNdb('original', imdb_ar, 12, true);

p1 = NNPatch(33, 33, [1 1]);
p2 = NNPatch(33, 33, [34 1]);
sel = [];
sel.tr_col_indices = [1:2 4]; %[1 2 4]; 
sel.nnpatches = [p1 p2];
[nndbs_tr, ~, ~, ~] = DbSlice.slice(nndb, sel);
           


            

% Import classes required for NNdb
import nnf.db.NNdb;
import nnf.db.DbSlice;

% Import all algorithms in alg package
import nnf.alg.*;

% Create a NNdb database with AR database (12 images per identity)
nndb = NNdb('original', imdb_ar, 12, true);
sel.tr_col_indices        = [1:3 7:12]; % [1 2 3 7 8 9 10 11 12]; 
sel.te_col_indices        = [4:6]; % [4 5 6];
sel.use_rgb               = false;              
sel.scale                 = 0.5;
sel.histeq                = true;
sel.class_range           = [1:36 61:76 78:100];
sel.vectorize_db          = false;
[nndb_tr, ~, nndb_te, ~]  = DbSlice.slice(nndb, sel); 

% help DbSlice.examples % For extensive help on Db slicing

%% PCA_L2
import nnf.alg.PCA;
info = [];
W = PCA.l2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

% PCA_MCC
import nnf.alg.PCA;
info = [];
W = PCA.mcc(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

% PCA-MCC (old implementation) - deprecated
% features_tr = reshape(nndb_tr.db, nndb_tr.h * nndb_tr.w * nndb_tr.ch, nndb_tr.n);
% features_te = reshape(nndb_te.db, nndb_tr.h * nndb_tr.w * nndb_tr.ch, nndb_te.n);
% MPCA(double(features_tr)/255, unique(nndb_tr.n_per_class), double(features_te)/255, nndb_te.cls_lbl);

%% KPCA
import nnf.alg.KPCA;
info = [];
for i=[100:10:360]
    info.t = i;
    [W, ki] = KPCA.l2(nndb_tr, info);
    accurary = Util.test(W, nndb_tr, nndb_te, ki)
end

%% LDA_L2 Fisher Faces
import nnf.alg.LDA;
info = [];
W = LDA.fl2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

% Direct LDA
import nnf.alg.LDA;
info = [];
[W, info] = LDA.dl2(nndb_tr, info);
accurary = Util.test(W, nndb_tr, nndb_te)

% LDA_L2 Regularized
import nnf.alg.LDA;
info = [];
W = LDA.l2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

% R1/L1 LDA
import nnf.alg.LDA;
info = [];
W = LDA.r1(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

% R1/L1 LDA (old implementation) - deprecated
% features_tr = reshape(nndb_tr.db, nndb_tr.h * nndb_tr.w * nndb_tr.ch, nndb_tr.n);
% features_te = reshape(nndb_te.db, nndb_tr.h * nndb_tr.w * nndb_tr.ch, nndb_te.n);
% LDA_L1(double(features_tr)/255, double(unique(nndb_tr.n_per_class)), nndb_tr.cls_lbl, double(features_te)/255, nndb_te.cls_lbl);

%% KDA_L2 SVD
import nnf.alg.KDA;
info = [];
for i=[6000:10:15000]
    info.t = i;
    [W, ki] = KDA.l2(nndb_tr, info);
    accurary = Util.test(W, nndb_tr, nndb_te, ki)
end

% KDA_L2 Regularized
import nnf.alg.KDA;
info = [];
info.Regu = true;
for i=[100:10:360]
    info.t = i;
    [W, ki] = KDA.l2(nndb_tr, info);
    accurary = Util.test(W, nndb_tr, nndb_te, ki)
end

%% SRC
import nnf.alg.SRC;
info = [];
SRC.l1(nndb_tr, nndb_te, info);

%% High Resolution Database            
sel.scale                 = 1;
[nndb_tr0, ~, ~, ~]       = DbSlice.slice(nndb, sel); 

import nnf.alg.PLS;
info = [];
info.bases = 100;
[W_HR, W_LR] = PLS.l2(nndb_tr0, nndb_tr, info);
accurary = Util.PLS_test(W_HR, nndb_tr0, W_LR, nndb_tr, nndb_te);

%% LLE
import nnf.alg.LLE;
info = [];
info.ReducedDim = 2; % 2
nndb_lle = LLE.do(nndb_tr, info);
figure, nndb_lle.plot();
figure, nndb_lle.plot(18); % Visualize the images that belong to first two identities (9*2)
figure, nndb_lle.plot(18, 73*9+1); % Visualize the images that belong to last two identities

% LLE + LDA/PCA %%
import nnf.alg.LLE;
info = [];
sel.tr_col_indices = [1:12];
info.ReducedDim = 599;
nndb_tr = DbSlice.slice(nndb, sel);
nndb_lle = LLE.do(nndb_tr, info);

sel = [];
sel.tr_col_indices = [1 2 4 8 9 11];
sel.te_col_indices = [3 5 6 7 10 12];
[ nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb_lle, sel); 

import nnf.alg.LDA;
info = [];
W = LDA.fl2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

info.Regu = 1;
W = LDA.l2(nndb_tr, info);
accurary = Util.test(W, nndb_tr, nndb_te)

import nnf.alg.PCA;
W = PCA.l2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)


%% TSNE
import nnf.alg.TSNE;
info = [];
info.ReducedDim = 2;
info.initial_dims = 26;
info.perplexity = 5;
nndb_tsne = TSNE.do(nndb_tr, info);
figure, nndb_tsne.plot();
figure, nndb_tsne.plot(18); % Visualize the images that belong to first two identities (9*2)
figure, nndb_tsne.plot(18, 73*9+1); % Visualize the images that belong to last two identities

%% DCC
% Toy Example
info = [];
info.test_1D = true;
info.plot_LDA = true;
DCC.test_l2(info);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADVANCED EXAMPLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1: Perform LDA, project all training samples to the LDA space and visualize with TSNE.
%
import nnf.db.Format;
w_features = W' * nndb_tr.features;
nndb_lda = NNdb('LDA', w_features, nndb_tr.n_per_class, true, [], Format.H_N);
%nndb_lda = NNdb('LDA', w_features, nndb_tr.n_per_class, false, nndb_tr.cls_lbl, Format.H_N);
nndb_lda.plot();

import nnf.alg.TSNE;
info = [];
info.ReducedDim = 2;
info.initial_dims = 74;
info.perplexity = 5;
nndb_tsne = TSNE.do(nndb_lda, info);
nndb_tsne.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2: Perform LLE, then classfication algorithm for recognition.
%
import nnf.alg.LLE;

% Perform LLE on full dataset
info = [];
sel.tr_col_indices = [1:12];
info.ReducedDim = 599;
nndb_full = DbSlice.slice(nndb, sel);
nndb_lle = LLE.do(nndb_full, info);

% Slice the LLE dataset into training and testing
sel = [];
sel.tr_col_indices = [1 2 4 8 9 11];
sel.te_col_indices = [3 5 6 7 10 12];
[ nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb_lle, sel); 

% Perform classification
import nnf.alg.LDA;
info = [];
W = LDA.fl2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

info.Regu = 1;
W = LDA.l2(nndb_tr, info);
accurary = Util.test(W, nndb_tr, nndb_te)

import nnf.alg.PCA;
W = PCA.l2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2: Perform TSNE, then classfication algorithm for recognition.
%
import nnf.alg.TSNE;
info = [];
info.ReducedDim = 500;
info.initial_dims = 1199;
info.perplexity = 5;
info.max_iter = 500;
sel.tr_col_indices = [1:12];
nndb_full = DbSlice.slice(nndb, sel);
nndb_tsne = TSNE.do(nndb_full, info);

% Slice the TSNE dataset into training and testing
sel = [];
sel.tr_col_indices        = [1 2 4 8 9 11];
sel.te_col_indices        = [3 5 6 7 10 12];
[ nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb_tsne, sel); 

% Perform classification
import nnf.alg.LDA;
info = [];
W = LDA.l2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)

info.Regu = 1;
W = LDA.l2(nndb_tr, info);
accurary = Util.test(W, nndb_tr, nndb_te)

import nnf.alg.PCA;
W = PCA.l2(nndb_tr);
accurary = Util.test(W, nndb_tr, nndb_te)









