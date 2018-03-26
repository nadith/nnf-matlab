% Import classes required for data generation
import civil.shm.generators.SHM7SDatGenerator;
import civil.shm.generators.Mode;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Civil SHM Data Generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MEMORY MODE
shm = SHM7SDatGenerator('ME', 1, {[1]; [1 2]; [1 3]}, [0.95:-0.01:0.80]);
data = shm.generate();

% Special method. (will ignore the damage cases, specified via the constructor and 
% only consider single element damage cases) 
data = shm.generate_all_se();

% DISK MODE
shm = SHM7SDatGenerator('ME', Mode.DISK_MODE, {[1 2]; [1 3]}, [0.95:-0.01:0.80], 'DataSaveDirectory', 'd:/civil');
data = shm.generate();  % no data will be returned. But will be written to the disk

% MEMORY DISK MODE
shm = SHM7SDatGenerator('ME', Mode.MEMORY_DISK_MODE, {[1]; [1 3]}, [0.95:-0.01:0.80], 'DataSaveDirectory', 'd:/civil');
data = shm.generate();

%% Using `combinator` to generate damage cases
% All combinations for choice of 2 elements out of 70 (without repetition)
shm = SHM7SDatGenerator('SE1', Mode.MEMORY_MODE, num2cell(combinator(70, 2, 'c'), 2), [0.95:-0.01:0.93]);
data = shm.generate();

% All combinations for the 2 element damages in the left side of the structure
shm = SHM7SDatGenerator('SE1', Mode.MEMORY_MODE, num2cell(combinator(21, 2, 'c'), 2), [0.95:-0.01:0.93]);
data = shm.generate();


%%
% Adding noise in the generation of data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adding uncertainity
shm = SHM7SDatGenerator('ME', Mode.MEMORY_MODE, {[1 2]; [1 3]}, [0.95:-0.01:0.80], ...
    'UncertainityLevel', 0.01, 'UncertainityPerSample', 5);
data = shm.generate();

% Adding measurement noise
shm = SHM7SDatGenerator('ME', Mode.MEMORY_MODE, {[1 2]; [1 3]}, [0.95:-0.01:0.80], ...
    'MeasurementNoiseOnFreq', 0.01, 'MeasurementNoiseOnModeShape', 0.05, 'MeasurementNoisePerSample', 5);
data = shm.generate();

% Adding both noise types
shm = SHM7SDatGenerator('ME', Mode.MEMORY_MODE, {[1 2]; [1 3]}, [0.95:-0.01:0.80], ...
    'UncertainityLevel', 0.01, 'UncertainityPerSample', 5, ...
    'MeasurementNoiseOnFreq', 0.01, 'MeasurementNoiseOnModeShape', 0.05, 'MeasurementNoisePerSample', 5);
data = shm.generate();

% Adding both noise types
shm = SHM7SDatGenerator('ME', Mode.DISK_MODE, {[1 2]; [1 3]}, [0.95:-0.01:0.80], ...
    'DataSaveDirectory', 'd:/civil', ...
    'UncertainityLevel', 0.01, 'UncertainityPerSample', 5, ...
    'MeasurementNoiseOnFreq', 0.01, 'MeasurementNoiseOnModeShape', 0.05, 'MeasurementNoisePerSample', 5);
data = shm.generate();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADVANCED EXAMPLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Big data generation exclusive example %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Genration for 7 storey building
import civil.shm.generators.SHM7SDatGenerator;
import civil.shm.generators.Mode;

damage_cases = {[2]; [23]; [5]; [27]; [8]; [31]; [11]; [35]; [14]; [39]; [17]; [43]; [20]; [47]; ...
                [2 23]; [5 27]; [8 31]; [11 35]; [14 39]; [17 43]; [20 47]}; 

shm = SHM7SDatGenerator('ME', Mode.DISK_MODE,  damage_cases, [0.95:-0.01:0.80], ...
    'DataSaveDirectory', 'E:/#JUN_WORKSPACE/7S DATA/civil', ...
    'UncertainityLevel', 0.01, 'UncertainityPerSample', 20, ...
    'MeasurementNoiseOnFreq', 0.01, 'MeasurementNoiseOnModeShape', 0.05, 'MeasurementNoisePerSample', 20);
data = shm.generate();

% Additional Function: If need to combine the generated data files above into a one file
SHM7SDatGenerator.combine_files({ ...
        'E:/#JUN_WORKSPACE/7S DATA/civil/ME_0.mat', ...
        'E:/#JUN_WORKSPACE/7S DATA/civil/ME_16.mat', ...
        'E:/#JUN_WORKSPACE/7S DATA/civil/ME_32.mat', ...
        'E:/#JUN_WORKSPACE/7S DATA/civil/ME_48.mat', ...
        'E:/#JUN_WORKSPACE/7S DATA/civil/ME_64.mat', ...
        }, 'COMBINED_PP.mat');
            
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize moving window preprocessor to perform feature generation via moving window process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% For classification problems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1. Basic usage
%
import nnf.pp.MovingWindowPreProcessor;
import civil.shm.generators.mwpp_load_shm_data;

pp_img_folder = 'E:/#JUN_WORKSPACE/7S DATA/civil_images_test';
pp = MovingWindowPreProcessor('E:/#JUN_WORKSPACE/7S DATA/civil_test', @mwpp_load_shm_data, pp_img_folder, ...
                                'SplitRatios', [0.8 0.10 0.10], ...
                                'TargetImageSize', [227 227]);
wsizes = [20:20:200];
steps = (400 - wsizes)/20;
pp.init(steps, wsizes);
[tr_table, val_table, te_table] = pp.preprocess();

% For regression problems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1. Basic usage
%
import nnf.pp.MovingWindowPreProcessor;
import civil.shm.generators.mwpp_load_shm_data;

pp_img_folder = 'E:/#JUN_WORKSPACE/7S DATA/civil_images_test2_2';
pp = MovingWindowPreProcessor('E:/#JUN_WORKSPACE/7S DATA/civil_test', @mwpp_load_shm_data, pp_img_folder, ...
                                'SplitRatios', [0.8 0.10 0.10], ...
                                'OutputNodes', 70, ...  # Output nodes for regression problems
                                'TargetImageSize', [227 227]);
wsizes = [20:20:200];
steps = (400 - wsizes)/20;
pp.init(steps, wsizes);
[tr_table, val_table, te_table] = pp.preprocess();


%
% 2. Utilizing stream data preprocessors
%
import nnf.pp.MovingWindowPreProcessor;
import civil.shm.generators.mwpp_load_shm_data;

pp_img_folder = 'E:/#JUN_WORKSPACE/7S DATA/civil_images_test2_2';
pp = MovingWindowPreProcessor('E:/#JUN_WORKSPACE/7S DATA/civil_test', @mwpp_load_shm_data, pp_img_folder, ...
                                'SplitRatios', [0.8 0.10 0.10], ...
                                'OutputNodes', 70, ...
                                'TargetImageSize', [227 227]);
wsizes = [20:20:200];
steps = (400 - wsizes)/20;
pp.init(steps, wsizes);

import nnf.pp.SDPP_MinMax;
import nnf.pp.SDPP_ZScore;
sdpp_minmax = SDPP_MinMax();
sdpp_zscore = SDPP_ZScore();
pipeline_id = pp.add_to_fit_pipeline(MovingWindowPreProcessor.PIPELINE_NEW, sdpp_minmax);
pp.add_to_fit_pipeline(pipeline_id, sdpp_zscore);
pp.add_to_standardize_pipeline(sdpp_zscore);

[tr_table, val_table, te_table] = pp.preprocess();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save generated tables in the disk
save('E:/#JUN_WORKSPACE/7S DATA/civil_images_test2.mat','tr_table', 'val_table', 'te_table', '-v7.3');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training a model with data generated above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. Train with a pre-trained CNN `AlexNet` for a classification task
tr_options = trainingOptions('sgdm',...
    'MiniBatchSize', 64,...
    'MaxEpochs', 4000,...
    'InitialLearnRate', 1e-4,...
    'Verbose', true,...      
    'Plots','training-progress',...
    'ValidationData', val_table,...
    'ValidationFrequency', 100,...
    'ValidationPatience', Inf,...
    'Shuffle', 'never');

cnet = CNNNet(tr_options, [], 'UseAlexNet', true, 'NumberOfClasses', 3);
tr_datasource = augmentedImageSource([227 227], tr_table, 'ColorPreprocessing', 'gray2rgb')
cnet = cnet.train_datasource(tr_datasource); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Train with a pre-trained CNN `AlexNet` for a regression task
%       Using a datasource for validation dataset via a `reporter` callback object (Custom extension).
%       IMPORTANT: Matlab has no support for reading from a datasource for validation dataset.
%
import nnf.alg.CNNNet

tr_options = trainingOptions('sgdm',...
    'MiniBatchSize', 64,...
    'MaxEpochs', 4000,...
    'InitialLearnRate',1e-4,...
    'Verbose', true,...      
    'Plots','training-progress',...
    'L2Regularization', 0.01, ...,  % 'ValidationData', val_table,...
    'ValidationFrequency', 500,...
    'ValidationPatience', Inf,...
    'Shuffle', 'never');            % => MUST BE 'Shuffle', 'never', TODO: Fix it
% 0.0099 |

cnet = CNNNet(tr_options, [], 'UseAlexNet', true, 'OutputNodes', 70);

% Create a datasource for training/validation data
tr_datasource = FilePathTableMiniBatchDatasourceEx(tr_table, fullfile(pp_img_folder, 'Target.mat'), 'TrOutput', [227 227], 'ColorPreprocessing', 'gray2rgb')

% Create a datasource for training/validation data
val_datasource = FilePathTableMiniBatchDatasourceEx(val_table, fullfile(pp_img_folder, 'Target.mat'), 'ValOutput', [227 227], 'ColorPreprocessing', 'gray2rgb')

% Create a datasource for validation data and a reporter object for callbacks during training
reporter = ValidationReporter(tr_options, val_datasource);

% Train with a training datasource and proces callbacks during training via `reporter`
cnet = cnet.train_datasource(tr_datasource, reporter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Train with a pre-trained CNN `AlexNet` for a regression task
%       Using a table for validation dataset (Matlab's default support).
%       IMPORTANT: Matlab has no support for reading from a datasource for validation dataset.
%
import nnf.alg.CNNNet

tr_options = trainingOptions('sgdm',...
    'MiniBatchSize', 64,...
    'MaxEpochs', 4000,...
    'InitialLearnRate',1e-4,...
    'Verbose', true,...      
    'Plots','training-progress',...
    'L2Regularization', 0.01, ...
    'ValidationData', val_table,... % Using a table for validation dataset
    'ValidationFrequency', 500,...
    'ValidationPatience', Inf,...    
    'Shuffle', 'never');
% 0.0099 |

% Initialize a CNNNet (using a pretrained model `AlexNet`)
cnet = CNNNet(tr_options, [], 'UseAlexNet', true, 'OutputNodes', 70);

% Create a datasource for training data
tr_datasource = FilePathTableMiniBatchDatasourceEx(...
                    tr_table, ...
                    fullfile(pp_img_folder, 'Target.mat'), ...
                    'TrOutput', [227 227], ...
                    'ColorPreprocessing', 'gray2rgb');
                
% Train with a training datasource (ValidationData: <table> specified in training options)
cnet = cnet.train_datasource(tr_datasource); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.1. Train with a custom made `AlexNet` CNN for a regression task
%
import nnf.alg.CNNNet

tr_options = trainingOptions('sgdm',...
    'MiniBatchSize', 64,...
    'MaxEpochs', 4000,...
    'InitialLearnRate',1e-4,...
    'Verbose', true,...      
    'Plots','training-progress',...
    'L2Regularization', 0.01, ...
    'ValidationData', val_table,...
    'ValidationFrequency', 500,...
    'ValidationPatience', Inf,... 
    'Shuffle', 'never');
% 0.0099 |

% Initialize a CNNNet (using a custom `AlexNet` model)
tmp = alexnet;
layerDefinitions = tmp.Layers(1:end-8);
layerDefinitions = [layerDefinitions
                    tanhLayer
                    fullyConnectedLayer(70, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)                    
                    regressionLayer];        
cnet = CNNNet(tr_options, layerDefinitions);

% Create a datasource for training data
tr_datasource = FilePathTableMiniBatchDatasourceEx(...
                    tr_table, ...
                    fullfile(pp_img_folder, 'Target.mat'), ...
                    'TrOutput', [227 227], ...
                    'ColorPreprocessing', 'gray2rgb');
                
% Train with a training datasource
cnet = cnet.train_datasource(tr_datasource); 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluating the network
pTe = predict(cnet, val_table(:, 1))';
aTe = table2array(val_table(:, 2:end))';
r_tr = regression(pTe, aTe, 'one')
scatter(aTe(:), pTe(:))
output = [];
output(:, 1) = predict(cnet, val_table(500, 1))';
output(:, 2) = table2array(val_table(500, 2:end))';