% Import classes required for NNdb
import nnf.db.NNdb;
import nnf.db.Format;
import nnf.db.DbSlice;
import nnf.db.Selection;

% Import all algorithms in alg package
import nnf.alg.*;

% Import illumination normalization pre-processing function
import nnf.utl.illumination_norm;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NNDB OPERATIONS, SLICING, ALGORITHM USAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a NNdb database with AR database (12 images per identity)
nndb = NNdb('original', imdb_ar, 12, true);

% % Augmenting dataset
% import nnf.utl.ImageAugment;
% info.center_cls_mean = true;
% info.samples_per_class = 2;
% info.noise_ratio = 0.2;
% nndb = ImageAugment.gauss_data_gen(nndb, info, true);
% nndb.show(10, 14);

% Basic Selection
sel = Selection();
sel.tr_col_indices        = [1:8];              % randperm(12, 8); % Random choice
sel.te_col_indices        = [9:12];             % setdiff([1:12], sel.tr_col_indices);
sel.use_rgb               = false;              
sel.scale                 = 1;                  % or [66 66]
sel.histeq                = true;
sel.class_range           = [1:100];
% sel.pre_process_script  = @illumination_norm; % Refer nnf.utl.illumination_norm
[nndb_tr, ~, nndb_te, ~, ~, ~, ~] = DbSlice.slice(nndb, sel); 
nndb_tr.show(10, 8)
figure, nndb_te.show(10, 4)
figure, nndb_tr.show_ws(10, 8) % with whitespaces
% help DbSlice.examples % For extensive help on Db slicing

%% PCA_L2
import nnf.alg.PCA;
info = [];
W = PCA.l2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

% PCA_MCC
import nnf.alg.PCA;
info = [];
W = PCA.mcc(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

% PCA-MCC (old implementation) - deprecated
% features_tr = reshape(nndb_tr.db, nndb_tr.h * nndb_tr.w * nndb_tr.ch, nndb_tr.n);
% features_te = reshape(nndb_te.db, nndb_tr.h * nndb_tr.w * nndb_tr.ch, nndb_te.n);
% MPCA(double(features_tr)/255, unique(nndb_tr.n_per_class), double(features_te)/255, nndb_te.cls_lbl);

% PCA Reconstruction (Occluded images)
import nnf.alg.PCA;
info = [];
[W, m] = PCA.l2(nndb_tr);
info.oc_filter.percentage = 0.5;
info.oc_filter.type = 'b'; % ('t':top, 'b':bottom, 'l':left, 'r':right) occlusion
nndb_prob_r = PCA.l2_reconst(W, m, nndb_te, info); % Reconstruct

%% KPCA
import nnf.alg.KPCA;
info = [];
for i=[100:10:360]
    info.t = i;
    [W, ki] = KPCA.l2(nndb_tr, info);
    accuracy = Util.test(W, nndb_tr, nndb_te, ki)
end

%% LDA_L2 Fisher Faces
import nnf.alg.LDA;
info = [];
W = LDA.fl2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

% Direct LDA
import nnf.alg.LDA;
info = [];
[W, info] = LDA.dl2(nndb_tr, info);
accuracy = Util.test(W, nndb_tr, nndb_te)

% LDA_L2 Regularized
import nnf.alg.LDA;
info = [];
W = LDA.l2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

% R1/L1 LDA
import nnf.alg.LDA;
info = [];
W = LDA.r1(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

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
    accuracy = Util.test(W, nndb_tr, nndb_te, ki)
end

% KDA_L2 Regularized
import nnf.alg.KDA;
info = [];
info.Regu = true;
for i=[100:10:360]
    info.t = i;
    [W, ki] = KDA.l2(nndb_tr, info);
    accuracy = Util.test(W, nndb_tr, nndb_te, ki)
end

%% SRC
import nnf.alg.SRC;
info = []; % - ADMM solver by default
info.mean_diff = false;
[accuracy, ~, sinfo] = SRC.l1(nndb_tr, nndb_te, info);

import nnf.alg.SRC;
info = [];
info.mean_diff = true;
info.method.name = 'L1BENCHMARK.FISTA';
info.method.param.tolerance = 0.0001;
[accuracy, ~, sinfo] = SRC.l1(nndb_tr, nndb_te, info);

import nnf.alg.SRC;
info = [];
info.mean_diff = true;
info.method.name = 'L1BENCHMARK.L1LS';
[accuracy, ~, sinfo] = SRC.l1(nndb_tr, nndb_te, info);

import nnf.alg.SRC;
info= [];
info.mean_diff = true;
info.method.name = 'SRV1_9.SRC2.interiorPoint';
[accuracy, ~, sinfo] = SRC.l1(nndb_tr, nndb_te, info);

import nnf.alg.SRC;
info = [];
info.mean_diff = true;
info.method.name = 'SRV1_9.SRC2.activeSet';
[accuracy, ~, sinfo] = SRC.l1(nndb_tr, nndb_te, info);

% Alternative to perform classification with residual in SRC
import nnf.alg.Util;
[accuracy2] = Util.src_test(nndb_tr, nndb_te, sinfo);
assert(accuracy == accuracy2);

%% High Resolution Database, PLS
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

sel = Selection();
sel.tr_col_indices = [1 2 4 8 9 11];
sel.te_col_indices = [3 5 6 7 10 12];
[ nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb_lle, sel); 

import nnf.alg.LDA;
info = [];
W = LDA.fl2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

info.Regu = 1;
W = LDA.l2(nndb_tr, info);
accuracy = Util.test(W, nndb_tr, nndb_te)

import nnf.alg.PCA;
W = PCA.l2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)


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
% Example 0: Face alignment and cropping
% % Desired Face: Default values
% info.desired_face.height = 66;     
% info.desired_face.width = 66;  
% info.desired_face.lex_ratio = 0.25;
% info.desired_face.rex_ratio = 0.75; 
% info.desired_face.ey_ratio = 0.33; 
% info.desired_face.my_ratio = 0.63;

% % Face Detector: Default values
% info.detector.show_im_fail = true;
% info.detector.vj.MinNeighbors = 4;    % Minimum neighbours for viola jones (Matlab)
% info.detector.vj.ScaleFactor = 1.1;   % Scale factor for viola jones (Matlab)
% info.detector.vj.MinSize = [20 20];   % Minimum size for viola jones (Matlab)

% % If filtering needed: Filter frontal pose config
% info.filter.pose_pitch = [-8 8];
% info.filter.pose_yaw = [-15 15];
% info.filter.pose_roll = [-5 5];

% If saving required
info.save_path = 'C:\Aligned_DB';

% Take images from a directory in the disk. (each class has a folder)
[nndb_aln, nndb_fail, fail_cls_lbl] = FaceAligner.align('C:\ImageDB', false, info);

% Take images from a NNdb in memory
[nndb_aln, nndb_fail, fail_cls_lbl] = FaceAligner.align(nndb, true, info);

% Visualize
figure, nndb_aln.show(10, 10)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1: Select randomly unique 3 images from each class and save it to a database file
import nnf.utl.rand_unq
cell_indices = cell(0, 0);
cls_count = 100;
for i=1:cls_count
    cell_indices{i} = rand_unq(3, 12);
end
sel = Selection();
sel.tr_col_indices        = cell_indices;
sel.class_range           = [1:cls_count];
[nndb_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
figure, nndb_tr.show(10, 3);

% Example 1.1: Save methods
nndb_tr.save('IMDB_66_66_AR_3.mat');                    % save the mat file
nndb_tr.save_compressed('IMDB_66_66_AR_3_comp.mat');    % save the mat file (reduced size)
nndb_tr.save_dir('C:\ImageDB');                         % creates folder for each class
nndb_tr.save_dir('C:\ImageDB', false);                  % save all images in single folder

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2: Perform LDA, project all training samples to the LDA space and visualize with TSNE.
%
w_features = W' * nndb_tr.features;
nndb_lda = NNdb('LDA', w_features, nndb_tr.n_per_class, true, [], Format.H_N);
%nndb_lda = NNdb('LDA', w_features, nndb_tr.n_per_class, false, nndb_tr.cls_lbl, Format.H_N);
%nndb_lda.plot();

import nnf.alg.TSNE;
info = [];
info.ReducedDim = 2;
info.initial_dims = 74;
info.perplexity = 5;
nndb_tsne = TSNE.do(nndb_lda, info);
nndb_tsne.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 3: Perform LLE, then classfication algorithm for recognition.
%
import nnf.alg.LLE;

% Perform LLE on full dataset
info = [];
sel.tr_col_indices = [1:12];
info.ReducedDim = 599;
nndb_full = DbSlice.slice(nndb, sel);
nndb_lle = LLE.do(nndb_full, info);

% Slice the LLE dataset into training and testing
sel = Selection();
sel.tr_col_indices = [1 2 4 8 9 11];
sel.te_col_indices = [3 5 6 7 10 12];
[ nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb_lle, sel); 

% Perform classification
import nnf.alg.LDA;
info = [];
W = LDA.fl2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

info.Regu = 1;
W = LDA.l2(nndb_tr, info);
accuracy = Util.test(W, nndb_tr, nndb_te)

import nnf.alg.PCA;
W = PCA.l2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 4: Perform TSNE, then classfication algorithm for recognition.
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
sel = Selection();
sel.tr_col_indices        = [1 2 4 8 9 11];
sel.te_col_indices        = [3 5 6 7 10 12];
[ nndb_tr, ~, nndb_te, ~] = DbSlice.slice(nndb_tsne, sel); 

% Perform classification
import nnf.alg.LDA;
info = [];
W = LDA.l2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

info.Regu = 1;
W = LDA.l2(nndb_tr, info);
accuracy = Util.test(W, nndb_tr, nndb_te)

import nnf.alg.PCA;
W = PCA.l2(nndb_tr);
accuracy = Util.test(W, nndb_tr, nndb_te)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 5: Structured SRC (Ref: Robust face recognition via occlusion dictionary learning)
%
% For Scarf occlusion
sel = Selection();
sel.tr_col_indices        = [1:6];
% sel.histeq                = true;
sel.val_col_indices       = [9];
sel.te_col_indices        = [10];
sel.use_rgb               = false;
sel.scale                 = [30 30];
sel.class_range           = [1:76 78:100]; %[1:75 77:100];  %[1:76 78:100];, 77 identity has a total black image
[nndb_A0, nndb_Ad_Y, nndb_Y, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
nndb_A0.show(10, 6);
figure, nndb_Ad_Y.show(10, 1);
figure, nndb_Y.show(10, 1);

% Learn a dictionary to represent scarf occlusion
import nnf.alg.SSRC;
info=[];
info.max_iter = 200;
info.visualize = true;
nndb_Ad = SSRC.dict_learn_proj_err(nndb_Ad_Y.zero_to_one, nndb_A0.zero_to_one, info);

% Learn coefficients x1, x2 to solve
% y = A0 * x1 + Ad * x2 + e
info=[];
info.lambdas = [0.001 0.001];
info.max_iter = 200;
info.visualize = true;
[coeffs, nndb_reconst] = SSRC.l1(nndb_Y.zero_to_one, [nndb_A0.zero_to_one nndb_Ad], info);

% Peform LDA classification on target database vs the reconstructed via (A0 * x1 + Ad * x2)
import nnf.alg.LDA;
info = [];
W = LDA.l2(nndb_reconst);
accuracy = Util.test(W, nndb_reconst, nndb_Y)

% TODO: Even though the paper suggests following method for classification, 
% the reconstruction is not of high quality after (y - Ad * x2)
P_reconst = nndb_Y.zero_to_one.features - nndb_Ad.features * coeffs{2};
P_reconst = uint8(P_reconst.*255);
nndb_P_reconst = NNdb('P_reconst', reshape(P_reconst, nndb_Y.h, nndb_Y.w, nndb_Y.ch, []), nndb_Y.n_per_class, false, nndb_Y.cls_lbl);
nndb_P_reconst.show(10, 10)

sinfo = [];
sinfo.coeffs = coeffs{1};
[accuracy2] = Util.src_test(nndb_A0.zero_to_one, nndb_P_reconst, sinfo);
