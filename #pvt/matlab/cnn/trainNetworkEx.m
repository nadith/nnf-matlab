function [trainedNet, info] = trainNetworkEx(varargin)
% trainNetwork   Train a neural network
%
%   trainedNet = trainNetwork(ds, layers, options) trains and returns a
%   network trainedNet for a classification problem. ds is an
%   imageDatastore with categorical labels, layers is an array of network
%   layers or a LayerGraph and options is a set of training options.
%
%   trainedNet = trainNetwork(mbs, layers, options) trains and
%   returns a network trainedNet. mbs is an augmentedImageSource,
%   denoisingImageSource, or pixelLabelImageSource that specifies both
%   inputs and responses. Use augmentedImageSource to preprocess images
%   for deep learning, for example, to resize, rotate, and reflect input
%   images. Use denoisingImageSource to preprocess images for use in training
%   denoising networks. Use pixelLabelImageSource to specify inputs and
%   responses when training semantic segmentation networks.
%
%   trainedNet = trainNetwork(X, Y, layers, options) trains and returns a
%   network, trainedNet. The format for X depends on the input layer. For
%   an image input layer, X is a numeric array of images arranged so that
%   the first three dimensions are the width, height and channels, and the
%   last dimension indexes the individual images. In a classification
%   problem, Y specifies the labels for the images as a categorical vector.
%   In a regression problem, Y contains the responses arranged as a matrix
%   of size number of observations by number of responses, or a four
%   dimensional numeric array, where the last dimension corresponds to the
%   number of observations. 
%
%   trainedNet = trainNetwork(C, Y, layers, options) trains an LSTM network
%   for sequence-to-label and sequence-to-sequence classification problems.
%   C is a cell array containing sequence or time series predictors and Y
%   contains the categorical labels or categorical sequences. layers must
%   define an LSTM network. It must begin with a sequence input layer. The
%   entries of C are 2-D numeric arrays where the first dimension is the
%   number of values per timestep, and the second dimension is the length
%   of the sequence. For sequence-to-label problems, Y is a categorical
%   vector of labels. For sequence-to-sequence classification problems, Y
%   is a cell array of categorical sequences. The sequence lengths of the
%   response sequences must be identical to the sequence lengths of the
%   corresponding predictor sequences. For sequence-to-sequence problems
%   with one observation, C can be a 2-D matrix and Y can be a categorical
%   sequence.
%
%   trainedNet = trainNetwork(tbl, layers, options) trains and returns a
%   network, trainedNet. For networks with an image input layer, tbl is a
%   table containing predictors in the first column as either absolute or
%   relative image paths or images. Responses must be in the second column
%   as categorical labels for the images. In a regression problem,
%   responses must be in the second column as either vectors or cell arrays
%   containing 3-D arrays or in multiple columns as scalars. For networks
%   with a sequence input layer, tbl is a table containing absolute or
%   relative MAT file paths of predictors in the first column. For a
%   sequence-to-label classification problem, the second column must be a
%   categorical vector of labels. For a sequence-to-sequence classification
%   problem, the second column must be an absolute or relative file path to
%   a MAT file with a categorical sequence.
%
%   trainedNet = trainNetwork(tbl, responseName, ...) trains and returns a
%   network, trainedNet. responseName is a character vector specifying the
%   name of the variable in tbl that contains the responses.
%
%   trainedNet = trainNetwork(tbl, responseNames, ...) trains and returns a
%   network, trainedNet, for regression problems. responseNames is a cell
%   array of character vectors specifying the names of the variables in tbl
%   that contain the responses.
%
%   [trainedNet, info] = trainNetwork(...) trains and returns a network,
%   trainedNet. info contains information on training progress.
%
%   Example 1:
%       Train a convolutional neural network on some synthetic images
%       of handwritten digits. Then run the trained network on a test
%       set, and calculate the accuracy.
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [ ...
%           imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer()
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer()
%           classificationLayer()];
%       options = trainingOptions('sgdm');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 2:
%       Train a network on synthetic digit data, and measure its
%       accuracy:
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [
%           imageInputLayer([28 28 1], 'Name', 'input')
%           convolution2dLayer(5, 20, 'Name', 'conv_1')
%           reluLayer('Name', 'relu_1')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
%           reluLayer('Name', 'relu_2')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
%           reluLayer('Name', 'relu_3')
%           additionLayer(2,'Name', 'add')
%           fullyConnectedLayer(10, 'Name', 'fc')
%           softmaxLayer('Name', 'softmax')
%           classificationLayer('Name', 'classoutput')];
%
%       lgraph = layerGraph(layers);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
%
%       plot(lgraph);
%
%       options = trainingOptions('sgdm');
%       [net,info] = trainNetwork(XTrain, YTrain, lgraph, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   See also nnet.cnn.layer, trainingOptions, SeriesNetwork, DAGNetwork, LayerGraph.

%   Copyright 2015-2017 The MathWorks, Inc.

narginchk(3,5);

try
    [lgraph, opts, X, Y, reporter] = iParseInputArguments(varargin{:});
    haveDAGNetwork = iHaveDAGNetwork(lgraph);
    if haveDAGNetwork
        [trainedNet, info] = doTrainDAGNetworkEx(lgraph, opts, X, Y, reporter);
    else
        [trainedNet, info] = doTrainNetworkEx(lgraph, opts, X, Y, reporter);
    end
catch e
    iThrowCNNException( e );
end

end

function [trainedNet, info] = doTrainDAGNetworkEx(lgraph, opts, X, Y, ~)
haveDAGNetwork = true;

% Validate layer graph
iValidateLayerGraph( lgraph );

% Validate layers
iValidateLayers( lgraph.Layers );

% Infer layer graph parameters
lgraph = iInferParameters( lgraph, haveDAGNetwork );
layers = lgraph.Layers;

% Retrieve the internal layers
internalLayers = iGetInternalLayers(layers);

% Create an internal to external layers map
layersMap = iLayersMap( layers );

% Validate options
iValidateOptions( opts );

% Validate training data
iValidateTrainingDataForProblem( X, Y, layers );

% Set desired precision
precision = nnet.internal.cnn.util.Precision('single');

% Set up and validate parallel training
executionSettings = iSetupExecutionEnvironment( opts, haveDAGNetwork );

% Create a training dispatcher
trainingDispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision, ...
    executionSettings, layersMap.externalLayers( internalLayers ));


% Assert that the input data has a valid size for the network in use and
% the response size matches the output of the network. Rethrow exceptions
% as if they were thrown from the main function
iValidateTrainingDataSizeForNetwork(trainingDispatcher, lgraph, haveDAGNetwork);

% Initialize learnable parameters
internalLayers = iInitializeParameters(internalLayers, precision);

% Store labels into cross entropy layer or response names into mean-squared
% error layer
if iIsClassificationNetwork( internalLayers )
    classNames = trainingDispatcher.ClassNames;
    internalLayers = iMaybeStoreClassNames(internalLayers, classNames);
else
    responseNames = trainingDispatcher.ResponseNames;
    internalLayers = iStoreResponseNames(internalLayers, responseNames);
end

% Create the network
trainedNet = iCreateInternalNetwork( lgraph, internalLayers, haveDAGNetwork );

% Convert learnable parameters to the correct format
trainedNet = trainedNet.prepareNetworkForTraining( executionSettings );

% Create a validation dispatcher if validation data was passed in
validationDispatcher = iValidationDispatcher(opts, precision, layers, lgraph, haveDAGNetwork);

% Assert that training and validation data are consistent
iAssertTrainingAndValidationDispatcherHaveSameClasses( trainingDispatcher, validationDispatcher );

% Instantiate reporters as needed
networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(trainedNet);
[reporters, trainingPlotReporter] = iOptionalReporters(opts, internalLayers, layersMap, precision, executionSettings, networkInfo, trainingDispatcher, validationDispatcher, haveDAGNetwork);
errorState = nnet.internal.cnn.util.ErrorState();
cleanup = onCleanup(@()iFinalizePlot(trainingPlotReporter, errorState));

% Always create the info recorder (because we will reference it later) but
% only add it to the list of reporters if actually needed.
infoRecorder = iInfoRecorder( opts, internalLayers );
if nargout >= 2
    reporters.add( infoRecorder );
end

% Create a trainer to train the network with dispatcher and options
trainer = iCreateTrainer( opts, precision, reporters, executionSettings, haveDAGNetwork );

% Do pre-processing work required for normalizing data
trainedNet = trainer.initializeNetworkNormalizations(trainedNet, trainingDispatcher, precision, executionSettings, opts.Verbose);

% Do the training
trainedNet = trainer.train(trainedNet, trainingDispatcher);

% Do post-processing work (if any)
trainedNet = trainer.finalizeNetwork(trainedNet, trainingDispatcher);
trainedNet = trainedNet.prepareNetworkForPrediction();
trainedNet = trainedNet.setupNetworkForHostPrediction();
iComputeFinalValidationResultsForPlot(trainingPlotReporter, trainedNet);

% Return arguments
trainedNet = iCreateExternalNetwork(trainedNet, layersMap, haveDAGNetwork);
info = infoRecorder.Info;

% Update error state ready for the cleanup.
errorState.ErrorOccurred = false;
end

function [trainedNet, info] = doTrainNetworkEx(layers, opts, X, Y, reporter)
haveDAGNetwork = false;

% Validate layers
iValidateLayers( layers );

% Retrieve the internal layers
internalLayers = iGetInternalLayers(layers);

% Infer layers parameters and assert layers consistency
internalLayers = iInferParameters(internalLayers, haveDAGNetwork);

% Create an internal to external layers map
layersMap = iLayersMap( layers );

% Validate options
iValidateOptions( opts );

% Validate training data
iValidateTrainingDataForProblem( X, Y, layers );

% Set desired precision
precision = nnet.internal.cnn.util.Precision('single');

% Set up and validate parallel training
executionSettings = iSetupExecutionEnvironment( opts, haveDAGNetwork );

% Create a training dispatcher
trainingDispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision, ...
    executionSettings, layersMap.externalLayers( internalLayers ));


% Assert that the input data has a valid size for the network in use and
% the response size matches the output of the network. Rethrow exceptions
% as if they were thrown from the main function
iValidateTrainingDataSizeForNetwork(trainingDispatcher, internalLayers, haveDAGNetwork);

% Initialize learnable parameters
internalLayers = iInitializeParameters(internalLayers, precision);

% Store labels into cross entropy layer or response names into mean-squared
% error layer
if iIsClassificationNetwork( internalLayers )
    classNames = trainingDispatcher.ClassNames;
    internalLayers = iMaybeStoreClassNames(internalLayers, classNames);
else
    responseNames = trainingDispatcher.ResponseNames;
    internalLayers = iStoreResponseNames(internalLayers, responseNames);
end

% Create the network
trainedNet = iCreateInternalNetwork( layers, internalLayers, haveDAGNetwork );

% Convert learnable parameters to the correct format
trainedNet = trainedNet.prepareNetworkForTraining( executionSettings );

% Create a validation dispatcher if validation data was passed in
validationDispatcher = iValidationDispatcher(opts, precision, layers, internalLayers, haveDAGNetwork);

% Assert that training and validation data are consistent
iAssertTrainingAndValidationDispatcherHaveSameClasses( trainingDispatcher, validationDispatcher );

% Instantiate reporters as needed
networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(trainedNet);
[reporters, trainingPlotReporter] = iOptionalReporters(opts, internalLayers, layersMap, precision, executionSettings, networkInfo, trainingDispatcher, validationDispatcher, haveDAGNetwork);
errorState = nnet.internal.cnn.util.ErrorState();
cleanup = onCleanup(@()iFinalizePlot(trainingPlotReporter, errorState));

% Custom reporter
if ~isempty(reporter)
    reporters.add(reporter);
end

% Always create the info recorder (because we will reference it later) but
% only add it to the list of reporters if actually needed.
infoRecorder = iInfoRecorder( opts, internalLayers );
if nargout >= 2
    reporters.add( infoRecorder );
end

% Create a trainer to train the network with dispatcher and options
trainer = iCreateTrainer( opts, precision, reporters, executionSettings, haveDAGNetwork );

% Do pre-processing work required for normalizing data
trainedNet = trainer.initializeNetworkNormalizations(trainedNet, trainingDispatcher, precision, executionSettings, opts.Verbose);

% Do the training
trainedNet = trainer.train(trainedNet, trainingDispatcher);

% Do post-processing work (if any)
trainedNet = trainer.finalizeNetwork(trainedNet, trainingDispatcher);
trainedNet = trainedNet.prepareNetworkForPrediction();
trainedNet = trainedNet.setupNetworkForHostPrediction();
iComputeFinalValidationResultsForPlot(trainingPlotReporter, trainedNet);

% Return arguments
trainedNet = iCreateExternalNetwork(trainedNet, layersMap, haveDAGNetwork);
info = infoRecorder.Info;

% Update error state ready for the cleanup.
errorState.ErrorOccurred = false;
end

function [layers, opts, X, Y, reporter] = iParseInputArguments(varargin)
% iParseInputArguments   Parse input arguments of trainNetwork
%
% Output arguments:
%   layers  - An array of layers or a layer graph
%   opts    - An object containing training options
%   X       - Input data, this can be a data dispatcher, an image
%             datastore, a table, a numeric array or a cell array
%   Y       - Response data, this can be a numeric array or empty in case X
%             is a dispatcher, a table, an image datastore or a cell array

reporter = [];

X = varargin{1};
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAnImageDatastore( X )
    iAssertOnlyThreeArgumentsForIMDS( nargin );
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAMiniBatchDatasource( X)
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
    reporter = varargin{4};    
elseif iIsPixelLabelDatastore( X )
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif istable( X )
    secondArgument = varargin{2};
    if ischar(secondArgument) || iscellstr(secondArgument)
        % ResponseName syntax
        narginchk(4,4);
        responseNames = secondArgument;
        iAssertValidResponseNames(responseNames, X);
        X = iSelectResponsesFromTable( X, responseNames );
        Y = [];
        layers = varargin{3};
        opts = varargin{4};
    else
        narginchk(3,3);
        Y = [];
        layers = varargin{2};
        opts = varargin{3};
    end
elseif isnumeric( X )
    narginchk(4,5);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
    reporter = varargin{5};
elseif iscell( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
    reporter = varargin{5}; 
else
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidType'));
end
end

function [X, Y] = iGetValidationDataFromOptions( opts )
X = opts.ValidationData;
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
elseif iIsAnImageDatastore( X )
    Y = [];
elseif istable( X )
    Y = [];
elseif iscell( X )
    Y = X{2};
    X = X{1};
else
    % Do nothing. Invalid type is already checked when creating
    % trainingOptions
end
end

function iValidateLayers( layers )
% iValidateLayers   Assert that layers is a valid array of layers and each
% custom layer within is valid

if ~isa(layers, 'nnet.cnn.layer.Layer')
    error(message('nnet_cnn:trainNetwork:InvalidLayersArray'))
end

for ii=1:numel(layers)
    % If it's not a built-in layer, validate it
    if ~isa( layers(ii), 'nnet.internal.cnn.layer.Externalizable' )
        nnet.internal.cnn.layer.util.CustomLayerVerifier.validateMethodSignatures( layers(ii), ii );
    end
end

if iContainsLSTMLayers( layers )
    if iContainsRegressionLayers( layers )
        error(message('nnet_cnn:trainNetwork:RegressionNotSupportedForLSTM'))
    else
        tfArray = iContainsIncompatibleLayers( layers );
        if any( tfArray )
            incompatibleLayers = layers( tfArray );
            error(message('nnet_cnn:trainNetwork:IncompatibleLayers', ...
                class( incompatibleLayers(1) )))
        end
    end
end
end

function lgraph = iInferParameters( lgraph, haveDAGNetwork )
if haveDAGNetwork
    % lgraph is a LayerGraph.
    lgraph = inferParameters(lgraph);
else
    % lgraph is an array of internal layers.
    lgraph = nnet.internal.cnn.layer.util.inferParameters(lgraph);
end
end

function iValidateOptions( opts )
% iValidateOptions   Assert that opts is a valid training option object
if ~isa(opts, 'nnet.cnn.TrainingOptionsSGDM')
    error(message('nnet_cnn:trainNetwork:InvalidTrainingOptions'))
end
end

function internalLayers = iGetInternalLayers( layers )
internalLayers = nnet.internal.cnn.layer.util.ExternalInternalConverter.getInternalLayers( layers );
end

function iValidateTrainingDataForProblem( X, Y, layers )
% iValidateTrainingDataForProblem   Assert that the input training data X
% and response Y are valid for the class of problem considered
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataForProblem( X, Y, layers );
end

function iValidateTrainingDataSizeForNetwork(dispatcher, internalLayersOrLayerGraph, haveDAGNetwork)
% iValidateDataSizeForNetwork   Assert that the input training data has a
% valid size for the network in use and the response size matches the
% output of the network
trainingDataValidator = iTrainingDataValidator;
if haveDAGNetwork
    trainingDataValidator.validateDataSizeForDAGNetwork( dispatcher, internalLayersOrLayerGraph );
else
    trainingDataValidator.validateDataSizeForNetwork( dispatcher, internalLayersOrLayerGraph );
end
end

function iValidateValidationDataForProblem( X, Y, layers )
% iValidateValidationDataForProblem   Assert that the input validation data
% X and response Y are valid for the class of problem considered
iVerifyLayersForValidation( layers );
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataForProblem( X, Y, layers );
end

function iVerifyLayersForValidation( layers )
if iContainsLSTMLayers( layers )
    error(message('nnet_cnn:trainNetwork:ValidationNotSupportedForLSTM'));
end
end

function iValidateValidationDataSizeForNetwork(dispatcher, internalLayersOrLayerGraph, haveDAGNetwork)
% iValidateValidationDataSizeForNetwork   Assert that the input validation
% data has a valid size for the network in use and the response size
% matches the output of the network
validationDataValidator = iValidationDataValidator;
if haveDAGNetwork
    validationDataValidator.validateDataSizeForDAGNetwork( dispatcher, internalLayersOrLayerGraph );
else
    validationDataValidator.validateDataSizeForNetwork( dispatcher, internalLayersOrLayerGraph );
end
end

function iAssertTrainingAndValidationDispatcherHaveSameClasses( trainingDispatcher, validationDispatcher )
if ~isempty(validationDispatcher)
    if ~iHaveSameClassNames(trainingDispatcher, validationDispatcher)
        error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentClasses'));
    end
end
end

function tf = iHaveSameClassNames(trainingDispatcher, validationDispatcher)
% iHaveSameClassNames   Return true if the classes in trainingDispatcher
% have the same labels as the ones in validationDispatcher. This does not
% catch the situation in which one set is a subset of the other - that
% situation will be caught when we compare the number of classes in the
% datasets to the number of classes expected by the network
tf = all(ismember(trainingDispatcher.ClassNames, validationDispatcher.ClassNames));
end

function trainingDataValidator = iTrainingDataValidator()
trainingDataValidator = nnet.internal.cnn.util.TrainNetworkDataValidator( ...
    nnet.internal.cnn.util.TrainingDataErrorThrower );
end

function validationDataValidator = iValidationDataValidator()
validationDataValidator = nnet.internal.cnn.util.TrainNetworkDataValidator( ...
    nnet.internal.cnn.util.ValidationDataErrorThrower );
end

function iThrowCNNException( exception )
% Wrap exception in a CNNException, which reports the error in a custom way
err = nnet.internal.cnn.util.CNNException.hBuildCustomError( exception );
throwAsCaller(err);
end

function layers = iInitializeParameters(layers, precision)
for i = 1:numel(layers)
    layers{i} = layers{i}.initializeLearnableParameters(precision);
    if isa(layers{i}, 'nnet.internal.cnn.layer.LSTM' )
        layers{i} = layers{i}.initializeDynamicParameters(precision);
    end
end
end

function externalNetwork = iCreateExternalNetwork(internalNetwork, layersMap, haveDAGNetwork)
% Construct an External network. We assume by this stage you have called
% internalNet.prepareNetworkForPrediction() and
% internalNet.setupNetworkForHostPrediction().
if haveDAGNetwork
    externalNetwork = DAGNetwork(internalNetwork, layersMap);
else
    % SeriesNetwork has to be constructed from the internal layers not to lose
    % information about the internal custom layers
    externalNetwork = SeriesNetwork(internalNetwork.Layers, layersMap);
    % Reset the network state, so that if the network is recurrent it is
    % configured for prediction on an arbitrary mini-batch size
    externalNetwork = externalNetwork.resetState();
end
end

function externalNetwork = iPrepareAndCreateExternalNetwork(internalNetwork, layersMap, haveDAGNetwork)
% Prepare an internal network for prediction, then create an external
% network
internalNetwork = internalNetwork.prepareNetworkForPrediction();
internalNetwork = internalNetwork.setupNetworkForHostPrediction();
externalNetwork = iCreateExternalNetwork(internalNetwork, layersMap, haveDAGNetwork);
end

function iComputeFinalValidationResultsForPlot(trainingPlotReporter, trainedNet)
trainingPlotReporter.computeFinalValidationResults(trainedNet);
end

function layersMap = iLayersMap( layers )
layersMap = nnet.internal.cnn.layer.util.InternalExternalMap( layers );
end

function infoRecorder = iInfoRecorder( opts, internalLayers )
trainingInfoContent = iTrainingInfoContent( opts, internalLayers );
infoRecorder = nnet.internal.cnn.util.traininginfo.Recorder(trainingInfoContent);
end

function aContent = iTrainingInfoContent( opts, internalLayers )
isValidationSpecified = iIsValidationSpecified(opts);

if iIsClassificationNetwork(internalLayers)
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationContent;
    end
else
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.RegressionWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.RegressionContent;
    end
end
end

function tf = iIsClassificationNetwork(internalLayers)
tf = iIsClassificationLayer(internalLayers{end});
end

function tf = iIsClassificationLayer(internalLayer)
tf = isa(internalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function layers = iMaybeStoreClassNames(layers, dispatcherClassNames)
% Store class names from dispatcher if layer does not have any.
shouldSetClassNames = isempty(layers{end}.ClassNames);
if shouldSetClassNames
    layers = iStoreClassNames(layers, dispatcherClassNames);
else
    userSpecifiedClassNames = layers{end}.ClassNames;
    
    if iDispatcherAndUserClassNamesDoNotMatch(...
            dispatcherClassNames, userSpecifiedClassNames)
        error(message('nnet_cnn:trainNetwork:InvalidClassNames', numel(layers)));
    end
end
end

function TF = iDispatcherAndUserClassNamesDoNotMatch(...
    dispatcherClassNames, userClassNames)
% The names must match. Including the order of the name.
TF = ~isequal(dispatcherClassNames, userClassNames);
end

function layers = iStoreClassNames(layers, labels)
layers{end}.ClassNames = labels;
end

function layers = iStoreResponseNames(layers, responseNames)
layers{end}.ResponseNames = responseNames;
end

function iAssertOnlyThreeArgumentsForIMDS( nArgIn )
if nArgIn~=3
    error(message('nnet_cnn:trainNetwork:InvalidNarginWithImageDatastore'));
end
end

function tf = iIsADataDispatcher(X)
tf = isa(X, 'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsAMiniBatchDatasource(X)
tf = isa(X,'nnet.internal.cnn.MiniBatchDatasource');
end

function tf = iIsParallelExecutionEnvironment(executionEnvironment)
tf = ismember( executionEnvironment, {'multi-gpu', 'parallel'} );
end

function executionSettings = iSetupExecutionEnvironment( opts, haveDAGNetwork )
% Detect CPU/GPU/multiGPU/parallel training, and set up environment
% appropriately
executionSettings = struct( ...
    'executionEnvironment', 'cpu', ...
    'useParallel', false, ...
    'workerLoad', 1 );
isParallel = iIsParallelExecutionEnvironment( opts.ExecutionEnvironment );
if ( isParallel && haveDAGNetwork )
    error(message('nnet_cnn:trainNetwork:InvalidDAGExecutionEnvironment'));
end
if isParallel
    [executionSettings.useParallel, executionSettings.workerLoad] = ...
        iSetupAndValidateParallel( opts.ExecutionEnvironment, opts.WorkerLoad );
end

GPUShouldBeUsed = nnet.internal.cnn.util.GPUShouldBeUsed( ...
    opts.ExecutionEnvironment, executionSettings.workerLoad );
if GPUShouldBeUsed
    executionSettings.executionEnvironment = 'gpu';
end
end

function [useParallel, workerLoad] = iSetupAndValidateParallel( executionEnvironment, workerLoad )
% Pool and work-per-worker setup and validation
[useParallel, isMultiGpu, pool] = iValidateParallelPool( executionEnvironment );
if useParallel
    workerLoad = iValidateWorkerLoad( isMultiGpu, pool, workerLoad );
end
end

function [useParallel, isMultiGpu, pool] = iValidateParallelPool( executionEnvironment )
% Detect parallel training, open a pool if necessary, and validate that
% pool
useParallel = true;
pool = gcp('nocreate');

% Multi-GPU (local parallel pool)
% Expect a local pool to be open, or open one with one worker per GPU
isMultiGpu = false;
if string(executionEnvironment) == "multi-gpu"
    isMultiGpu = true;
    
    % Check that there are supported GPUs
    numGpus = gpuDeviceCount();
    if numGpus == 0
        error(message('parallel:gpu:device:NoCUDADevice'));
    end
    if ~isempty(pool)
        % Check that the open pool is local
        if ~isa( pool.Cluster, 'parallel.cluster.Local' )
            error(message('nnet_cnn:trainNetwork:ExpectedLocalPool'));
        end
    else
        % If no pool is open and there is only one supported GPU, we
        % should train as normal, on the client, without opening a
        % pool. User can force training to happen on a pool by opening
        % it themselves.
        if numGpus == 1
            isMultiGpu = false;
            useParallel = false;
            return;
        else
            % Check that the default cluster profile is local
            defaultProfileName = parallel.defaultClusterProfile();
            defaultProfileType = parallel.internal.settings.ProfileExpander.getClusterType( defaultProfileName );
            if defaultProfileType == parallel.internal.types.SchedulerType.Local
                % Open the default cluster with numGpus workers
                clust = parcluster( defaultProfileName );
                % Account for the possibility that user has changed the
                % default local profile to have fewer workers
                numGpus = min( numGpus, clust.NumWorkers );
                % Open pool. We need SPMD enabled and doing it when opening
                % the pool leads to faster communication.
                pool = parpool( clust, numGpus, 'SpmdEnabled', true );
            else
                error(message('nnet_cnn:trainNetwork:MultiGpuRequiresDefaultLocal', defaultProfileName));
            end
        end
    end
    
    % General parallel pool
    % Expect a pool to be open, or open the default pool
else
    if isempty(pool)
        % Error if user has disabled auto-pool creation
        s = settings;
        if s.parallel.client.pool.AutoCreate.ActiveValue == 0
            error(message('nnet_cnn:trainNetwork:ParallelAutoOpenDisabled'));
        end
        % Open pool using default profile
        pool = parpool( 'SpmdEnabled', true );
    end
end

% Check that SPMD is enabled in the current pool
if ~pool.SpmdEnabled
    error(message('nnet_cnn:trainNetwork:SPMDDisabled'));
end
end

function workerLoad = iValidateWorkerLoad( isMultiGpu, pool, userWorkerLoad )
% Given a parallel pool, modify the workerLoad settings to disable any
% workers that cannot access GPUs - unless there are no GPUs, in which case
% assume training on all pool CPUs

% Initialize workerLoad, using input user settings if provided
numWorkers = pool.NumWorkers;
if ~isempty(userWorkerLoad)
    % Validate user input
    if length(userWorkerLoad) ~= numWorkers
        error(message('nnet_cnn:trainNetwork:InvalidWorkerLoad'));
    end
else
    userWorkerLoad = ones( 1, numWorkers );
end
workerLoad = userWorkerLoad;

% Validate and mask workers that don't get their own GPU, if there
% are any GPUs
spmd
    [hostname, deviceIndex] = iGetHostnameAndDeviceIndex();
end
% Find unique hostname/device index combinations. Simple solution
% is to concatenate the two and use unique
deviceIndex = [ deviceIndex{:} ];
hostnameDevice = cell(numWorkers, 1);
maskDisabled = false(1, numWorkers);
for lab = 1:numWorkers
    % Eliminate workers that the user has disabled manually, as well as
    % workers on nodes with no GPUs
    if workerLoad(lab) == 0 || deviceIndex(lab) == 0
        hostnameDevice{lab} = 'disabled';
        maskDisabled(lab) = true;
    else
        hostnameDevice{lab} = sprintf('%s_%d', hostname{lab}, deviceIndex(lab));
    end
end
% Now reduce to the remaining unique combinations
[~, uniqueIndices] = unique(hostnameDevice, 'stable');
% Mask superfluous and no-GPU workers
mask = true( 1, numWorkers );
mask(uniqueIndices) = false;
mask(maskDisabled) = true;
% Apply mask to existing settings
workerLoad(mask) = 0;

% Special case: the result of the above is that all workers are idle, which
% can only happen if there are no workers with access to GPUs that haven't
% been disabled deliberately by the user - in this case we use all workers
% (that the user did not disable) and training will happen on the CPUs.
if all( workerLoad == 0 )
    workerLoad = userWorkerLoad;
    return;
end

% Report a warning if there are idle workers as a consequence of GPU
% sharing
numNewIdleWorkers = sum( (userWorkerLoad > 0) & (workerLoad == 0) );
if numNewIdleWorkers > 0
    if isMultiGpu
        warning(message('nnet_cnn:trainNetwork:SomeWorkersIdleLocal', numNewIdleWorkers));
    else
        warning(message('nnet_cnn:trainNetwork:SomeWorkersIdleCluster', numNewIdleWorkers));
    end
end

end

function dispatcher = iCreateTrainingDataDispatcher(X, Y, opts, precision, executionSettings, layers)
% Create a dispatcher.
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'discardLast', precision, executionSettings,...
    opts.Shuffle, opts.SequenceLength, opts.SequencePaddingValue, layers);
end

function dispatcher = iCreateValidationDataDispatcher(X, Y, opts, precision)
% iCreateValidationDataDispatcher   Create a dispatcher for validation data

% Validation dispatcher cannot be parallel
executionSettings.useParallel = false;
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'truncateLast', precision, executionSettings, opts.Shuffle, 'longest', 0);
end

function [reporter, trainingPlotReporter] = iOptionalReporters(opts, internalLayers, layersMap, precision, executionSettings, networkInfo, trainingDispatcher, validationDispatcher, haveDAGNetwork)
% iOptionalReporters   Create a vector of Reporters based on the given
% training options and the network type
%
% See also: nnet.internal.cnn.util.VectorReporter
reporter = nnet.internal.cnn.util.VectorReporter();

isValidationSpecified = iIsValidationSpecified(opts);

isAClassificationNetwork = iIsClassificationNetwork(internalLayers);
if opts.Verbose
    % If verbose is true, add a progress displayer
    if isAClassificationNetwork
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.ClassificationValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.ClassificationColumns;
        end
    else
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.RegressionValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.RegressionColumns;
        end
    end
    progressDisplayerFrequency = opts.VerboseFrequency;
    if isValidationSpecified
        progressDisplayerFrequency = [progressDisplayerFrequency opts.ValidationFrequency];
    end
    progressDisplayer = nnet.internal.cnn.util.ProgressDisplayer(columnStrategy);
    progressDisplayer.Frequency = progressDisplayerFrequency;
    reporter.add( progressDisplayer );
end

if isValidationSpecified
    % Create a validation reporter
    validationReporter = iValidationReporter( validationDispatcher, precision, executionSettings.executionEnvironment, opts.ValidationFrequency, opts.ValidationPatience, opts.Shuffle );
    reporter.add( validationReporter );
end

if ~isempty( opts.CheckpointPath )
    checkpointSaver = nnet.internal.cnn.util.CheckpointSaver( opts.CheckpointPath );
    checkpointSaver.ConvertorFcn = @(net)iPrepareAndCreateExternalNetwork(net, layersMap, haveDAGNetwork);
    reporter.add( checkpointSaver );
end

if ~isempty( opts.OutputFcn )
    userCallbackReporter = nnet.internal.cnn.util.UserCallbackReporter( opts.OutputFcn );
    reporter.add( userCallbackReporter );
end

if strcmp( opts.Plots, 'training-progress' )
    if isdeployed
        error(message('nnet_cnn:internal:cnn:ui:trainingplot:TrainingPlotNotDeployable'))
    end
    if ~isValidationSpecified
        validationReporter = nnet.internal.cnn.util.EmptyValidationReporter();   % To be used only by the trainingPlotReporter
    end
    trainingPlotReporter = iCreateTrainingPlotReporter(isAClassificationNetwork, executionSettings, opts, internalLayers, networkInfo, trainingDispatcher, isValidationSpecified, validationReporter);
    reporter.add( trainingPlotReporter );
else
    trainingPlotReporter = nnet.internal.cnn.util.EmptyPlotReporter();
end
end

function trainingPlotReporter = iCreateTrainingPlotReporter(isAClassificationNetwork, executionSettings, opts, internalLayers, networkInfo, trainingDispatcher, isValidationSpecified, validationReporter)
hasVariableNumItersPerEpoch = iHasVariableNumItersEachEpoch(opts, internalLayers);
if hasVariableNumItersPerEpoch
    epochDisplayer = nnet.internal.cnn.ui.axes.EpochDisplayHider();
    determinateProgress = nnet.internal.cnn.ui.progress.DeterminateProgressText();
    tableDataFactory = nnet.internal.cnn.ui.info.VariableEpochSizeTextTableDataFactory();
else
    epochDisplayer = nnet.internal.cnn.ui.axes.EpochAxesDisplayer();
    determinateProgress = nnet.internal.cnn.ui.progress.DeterminateProgressBar();
    tableDataFactory = nnet.internal.cnn.ui.info.TextTableDataFactory();
end

% create the view
legendLayout = nnet.internal.cnn.ui.layout.Legend();
textLayout = nnet.internal.cnn.ui.layout.TextTable();
trainingPlotView = nnet.internal.cnn.ui.TrainingPlotViewHG(determinateProgress, legendLayout, textLayout);

% create the presenter
if isAClassificationNetwork
    axesFactory = nnet.internal.cnn.ui.factory.ClassificationAxesFactory();
    metricRowDataFactory = nnet.internal.cnn.ui.info.ClassificationMetricRowDataFactory();
else
    axesFactory = nnet.internal.cnn.ui.factory.RegressionAxesFactory();
    metricRowDataFactory = nnet.internal.cnn.ui.info.RegressionMetricRowDataFactory();
end
executionInfo = nnet.internal.cnn.ui.ExecutionInfo(executionSettings.executionEnvironment, executionSettings.useParallel, opts.LearnRateScheduleSettings.Method, opts.InitialLearnRate);
validationInfo = nnet.internal.cnn.ui.ValidationInfo(isValidationSpecified, opts.ValidationFrequency, opts.ValidationPatience);
%
watch = nnet.internal.cnn.ui.adapter.Stopwatch();
stopReasonRowDataFactory = nnet.internal.cnn.ui.info.StopReasonRowDataFactory();
preprocessingDisplayer = iCreatePreprocessingDisplayer(networkInfo);
helpLauncher = nnet.internal.cnn.ui.info.TrainingPlotHelpLauncher();
epochInfo = iCreateEpochInfo(opts, trainingDispatcher);
dialogFactory = nnet.internal.cnn.ui.DialogFactory();
trainingPlotPresenter = nnet.internal.cnn.ui.TrainingPlotPresenterWithDialog(...
    trainingPlotView, tableDataFactory, metricRowDataFactory, stopReasonRowDataFactory, preprocessingDisplayer, dialogFactory, ...
    axesFactory, epochDisplayer, helpLauncher, watch, executionInfo, validationInfo, epochInfo);

% create the reporter
summaryFactory = nnet.internal.cnn.util.SummaryFactory();
trainingPlotReporter = nnet.internal.cnn.util.TrainingPlotReporter(trainingPlotPresenter, validationReporter, summaryFactory, epochInfo);
end

function iFinalizePlot(trainingPlotReporter, errorState)
trainingPlotReporter.finalizePlot(errorState.ErrorOccurred);
end

function epochInfo = iCreateEpochInfo(opts, trainingDispatcher)
epochInfo = nnet.internal.cnn.ui.EpochInfo(opts.MaxEpochs, trainingDispatcher.NumObservations, opts.MiniBatchSize);
end

function preprocessingDisplayer = iCreatePreprocessingDisplayer(networkInfo)
if networkInfo.ShouldImageNormalizationBeComputed
    dialogFactory = nnet.internal.cnn.ui.DialogFactory();
    preprocessingDisplayer = nnet.internal.cnn.ui.PreprocessingDisplayerDialog(dialogFactory);
else
    preprocessingDisplayer = nnet.internal.cnn.ui.PreprocessingDisplayerEmpty();
end
end

function tf = iHasVariableNumItersEachEpoch(opts, internalLayers)
isRNN = isa(internalLayers{1}, 'nnet.internal.cnn.layer.SequenceInput');
hasCustomSequenceLength = isnumeric(opts.SequenceLength);
tf = isRNN && hasCustomSequenceLength;
end

function validationDispatcher = iValidationDispatcher(opts, precision, layers, internalLayersOrLayerGraph, haveDAGNetwork)
% iValidationDispatcher   Get validation data and create a dispatcher for it. Validate the
% data for the current problem and w.r.t. the current architecture.

% Return empty if no validation data was specified
if ~iIsValidationSpecified(opts)
    validationDispatcher = [];
else
    % There is no need to convert datastore into table, since validation
    % will be computed only on one worker
    [XVal, YVal] = iGetValidationDataFromOptions( opts );
    iValidateValidationDataForProblem( XVal, YVal, layers );
    % Create a validation dispatcher
    validationDispatcher = iCreateValidationDataDispatcher(XVal, YVal, opts, precision);
    % Verify the dispatcher has valid size respect to input and output of
    % the network
    iValidateValidationDataSizeForNetwork(validationDispatcher, internalLayersOrLayerGraph, haveDAGNetwork);
end
end

function tf = iIsValidationSpecified(opts)
tf = ~isempty(opts.ValidationData);
end

function validator = iValidationReporter(validationDispatcher, precision, executionEnvironment, frequency, patience, shuffle)
validator = nnet.internal.cnn.util.ValidationReporter(validationDispatcher, precision, executionEnvironment, frequency, patience, shuffle);
end

function trainer = iCreateTrainer(opts, precision, reporters, executionSettings, haveDAGNetwork)
if haveDAGNetwork
    trainer = nnet.internal.cnn.DAGTrainer(opts, precision, reporters, executionSettings);
else
    if ~executionSettings.useParallel
        trainer = nnet.internal.cnn.Trainer(opts, precision, reporters, executionSettings);
    else
        trainer = nnet.internal.cnn.ParallelTrainer(opts, precision, reporters, executionSettings);
    end
end
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function iAssertValidResponseNames(responseNames, tbl)
% iAssertValidResponseNames   Assert that the response names are variables
% of the table and they do not refer to the first column.
variableNames = tbl.Properties.VariableNames;
refersToFirstColumn = ismember( variableNames(1), responseNames );
responseNamesAreAllVariables = all( ismember(responseNames,variableNames) );
if refersToFirstColumn || ~responseNamesAreAllVariables
    error(message('nnet_cnn:trainNetwork:InvalidResponseNames'))
end
end

function resTbl = iSelectResponsesFromTable(tbl, responseNames)
% iSelectResponsesFromTable   Return a new table with only the first column
% (predictors) and the variables specified in responseNames.
variableNames = tbl.Properties.VariableNames;
varTF = ismember(variableNames, responseNames);
% Make sure to select predictors (first column) as well
varTF(1) = 1;
resTbl = tbl(:,varTF);
end

function [hostid, deviceIndex] = iGetHostnameAndDeviceIndex()
hostid = parallel.internal.general.HostNameUtils.getLocalHostAddress();
try
    if nnet.internal.cnn.util.isGPUCompatible()
        deviceIndex = parallel.internal.gpu.currentDeviceIndex();
    else
        deviceIndex = 0;
    end
catch
    deviceIndex = 0;
end
end

function tf = iIsPixelLabelDatastore(x)
tf = isa(x, 'matlab.io.datastore.PixelLabelDatastore');
end

function iValidateLayerGraph(lgraph)
if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Invalid layer graph. A layer graph must be an object of type LayerGraph.');
end

isEmptyLayerGraph = size(lgraph.Connections,1) == 0;
if isEmptyLayerGraph
    error('Supplied layer graph cannot be empty.');
end
end

function internalNetwork = iCreateInternalNetwork( lgraph, internalLayers, haveDAGNetwork )
if haveDAGNetwork
    internalLayerGraph = iExternalToInternalLayerGraph( lgraph );
    internalLayerGraph.Layers = internalLayers;
    topologicalOrder = extractTopologicalOrder( lgraph );
    internalNetwork = nnet.internal.cnn.DAGNetwork(internalLayerGraph, topologicalOrder);
else
    internalNetwork = nnet.internal.cnn.SeriesNetwork(internalLayers);
end
end

function internalLayerGraph = iExternalToInternalLayerGraph( externalLayerGraph )
internalLayers = iGetInternalLayers( externalLayerGraph.Layers );
hiddenConnections = externalLayerGraph.HiddenConnections;
internalConnections = iHiddenToInternalConnections( hiddenConnections );
internalLayerGraph = nnet.internal.cnn.LayerGraph(internalLayers, internalConnections);
end

function internalConnections = iHiddenToInternalConnections( hiddenConnections )
internalConnections = nnet.internal.cnn.util.hiddenToInternalConnections( hiddenConnections );
end

function haveDAGNetwork = iHaveDAGNetwork(lgraph)
haveDAGNetwork = isa(lgraph,'nnet.cnn.LayerGraph');
end

function tf = iContainsLSTMLayers( layers )
tf = any( arrayfun( @(l)isa(l, 'nnet.cnn.layer.LSTMLayer'), layers ) );
end

function tf = iContainsRegressionLayers( layers )
tf = any( arrayfun( @(l)isa( l, 'nnet.cnn.layer.RegressionOutputLayer' ) ...
    || isa(l, 'nnet.layer.RegressionLayer' ), layers ) );
end

function tfArray = iContainsIncompatibleLayers( layers )
imageSpecificLayers = { ...
    'AveragePooling2DLayer';
    'BatchNormalizationLayer';
    'Convolution2DLayer';
    'CrossChannelNormalizationLayer';
    'ImageInputLayer';
    'MaxPooling2DLayer';
    'MaxUnpooling2DLayer';
    'TransposedConvolution2DLayer'; };
imageSpecificLayers = cellfun( @(s)['nnet.cnn.layer.' s], ...
    imageSpecificLayers, 'UniformOutput', false );
numLayers = numel( layers );
tfArray = false(numLayers, 1);
for ii = 1:numLayers
    tfArray(ii) = any( cellfun( @(t)isa( layers(ii), t ), imageSpecificLayers ) );
end
end
