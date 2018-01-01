classdef FilePathTableMiniBatchDatasourceEx <...
        nnet.internal.cnn.MiniBatchDatasource &...
        nnet.internal.cnn.NamedResponseMiniBatchDatasource &...
        nnet.internal.cnn.BackgroundDispatchableDatasource
    
    %  nnet.internal.cnn.DistributableFilePathTableMiniBatchDatasource &... TODO: Customize

    % FilePathTableMiniBatchDatasourceEx  class to extract 4D data from table data
    %
    % Input data    - a table containing predictors and responses. The
    %               first column will contain predictors holding file paths
    %               to the images. Responses will be held in the second
    %               column as either vectors or cell arrays containing 3-D
    %               arrays or categorical labels. Alternatively, responses
    %               will be held in multiple columns as scalars.
    % Output data   - 4D data where the fourth dimension is the number of
    %               observations in that mini batch.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Dependent)
       MiniBatchSize 
    end
    
    properties
        NumberOfObservations
        ResponseNames
        Datastore
    end
        
    properties (Access = ?nnet.internal.cnn.DistributableMiniBatchDatasource)
        % Table   (Table) The table containing all the data
        TableData
    end
    
    properties (Access = private)
        StartIndexOfCurrentMiniBatch 
        OrderedIndices % Shuffled sequence of all indices into observations
        MiniBatchSizeInternal
    end
    
    
    % Newly added properties
    properties (Access = private)
        outputFile_
        ImageAugmenter
        OutputRowsColsChannels % The expected output image size [numRows, numCols, numChannels].
    end
    
    properties (SetAccess = private)
        %DataAugmentation - Augmentation applied to input images
        %
        %    DataAugmentation is a scalar imageDataAugmenter object or a
        %    character vector or string. When DataAugmentation is 'none' 
        %    no augmentation is applied to input images.
        DataAugmentation
        
        %ColorPreprocessing - Pre-processing of input image color channels
        %
        %    ColorPreprocessing is a character vector or string specifying
        %    pre-proprocessing operations performed on color channels of
        %    input images. This property is used to ensure that all output
        %    images from the datasource have the number of color channels
        %    required by inputImageLayer. Valid values are
        %    'gray2rgb','rgb2gray', and 'none'. If an input images already
        %    has the desired number of color channels, no operation is
        %    performed. For example, if 'gray2rgb' is specified and an
        %    input image already has 3 channels, no operation is performed.
        ColorPreprocessing
        
        %OutputSize - Size of output images
        %
        %    OutputSize is a two element numeric vector of the form
        %    [numRows, numColumns] that specifies the size of output images
        %    returned by augmentedImageSource.
        OutputSize
        
        %OutputSizeMode - Method used to resize output images.
        %
        %    OutputSizeMode is a character vector or string specifying the
        %    method used to resize output images to the requested
        %    OutputSize. Valid values are 'centercrop', 'randcrop', and 
        %   'resize' (default).
        OutputSizeMode
    end
    
    properties
        outputFieldName    % Field to read data from
    end
    
    methods
        
        function self = FilePathTableMiniBatchDatasourceEx(tableIn, outputFilePath, outputFieldName, varargin)
            %   Parameters include:
            %
            %   'ColorPreprocessing'    A scalar string or character vector specifying
            %                           color channel pre-processing. This option can
            %                           be used when you have a training set that
            %                           contains both color and grayscale image data
            %                           and you need data created by the datasource to
            %                           be strictly color or grayscale. Options are:
            %                           'gray2rgb','rgb2gray','none'. For example, if
            %                           you need to train a network that expects color
            %                           images but some of the images in your training
            %                           set are grayscale, then specifying the option
            %                           'gray2rgb' will replicate the color channels of
            %                           the grayscale images in the input image set to
            %                           create MxNx3 output images.
            %
            %                           Default: 'none'
            %
            %   'DataAugmentation'      A scalar imageDataAugmenter object, string, or
            %                           character array that specifies
            %                           the kinds of image data augmentation that will
            %                           be applied to generated images.
            %
            %                           Default: 'none'
            %
            %   'OutputSizeMode'        A scalar string or character vector specifying the
            %                           technique used to adjust image sizes to the
            %                           specified 'OutputSize'. Options are: 'resize',
            %                           'centercrop', 'randcrop'.
            %
            %                           Default: 'resize'
            self.TableData = tableIn;
            
            if ~isempty(tableIn)
                self.Datastore = iCreateDatastoreFromTable(tableIn);
                % self.ResponseNames = iResponseNamesFromTable(tableIn);
                self.NumberOfObservations = size(tableIn,1);
                self.OrderedIndices = 1:self.NumberOfObservations;
                % self.MiniBatchSize = miniBatchSize;
                self.outputFieldName = outputFieldName;
                self.reset();
            else
                self = nnet.internal.cnn.FilePathTableMiniBatchDatasourceEx.empty();
            end
 
            self.outputFile_ = matfile(outputFilePath, 'Writable', false);
                        
            inputs = self.parseInputs(varargin{:});
            self.ImageAugmenter = inputs.DataAugmentation;
            
            self.determineExpectedOutputSize();
        end
                
        function [X,Y] = getObservations(self, indices)
            % getObservations  Overload of method to retrieve specific
            % observations.
            
            Y = self.readResponses(self.OrderedIndices(indices));
            
            % Create datastore partition via a copy and index. This is
            % faster than constructing a new datastore with the new
            % files.
            subds = copy(self.Datastore);
            subds.Files = self.Datastore.Files(indices);
            X = subds.readall();
           
            X = self.applyAugmentationPipelineToBatch(X);
        end
                
        function [X,Y] = nextBatch(self)
            % nextBatch  Return next mini-batch
            
            % Map the indices into data
            miniBatchIndices = self.computeDataIndices();
            
            % Read the data
            [X,Y] = self.readData(miniBatchIndices);
            
            % Advance indices of current mini batch
            self.advanceCurrentMiniBatchIndices();
        end
                    
        function reset(self)
            % Reset iterator state to first mini-batch
            
            self.StartIndexOfCurrentMiniBatch = 1;
            self.Datastore.reset();
        end
        
        function shuffle(self)
            % Shuffle  Shuffle the data
            
            self.OrderedIndices = randperm(self.NumberOfObservations);
            self.Datastore = iCreateDatastoreFromTable(self.TableData, self.OrderedIndices);
        end
        
        function reorder(self,indices)
            % reorder   Shuffle the data to a specific order
            
            self.OrderedIndices = indices;
            self.Datastore = iCreateDatastoreFromTable(self.TableData,self.OrderedIndices);
        end
        
        function set.MiniBatchSize(self,batchSize)
            value = min(batchSize,self.NumberOfObservations);
            self.MiniBatchSizeInternal = value;
            if ~isempty(self.TableData)
                self.Datastore.ReadSize = max(1,value);
            end
        end
        
        function batchSize = get.MiniBatchSize(self)
            batchSize = self.MiniBatchSizeInternal;
        end
        
    end
    
    methods (Access = private)
        
        function [X,Y] = readData(self,indices) 
            if isempty(self.TableData)
                [X,Y] = deal([]);  % Copy [] to X and Y
            else
                X = self.Datastore.read();
                Y = self.readResponses(indices);
            end
            
            X = self.applyAugmentationPipelineToBatch(X);                        
        end
        
        function response = readResponses(self, indices)
            
            % singleResponseColumn = size(self.TableData,2) == 2;
            % if singleResponseColumn
            %     response = self.TableData{indices,2};
            %     if isvector(self.TableData(1,2))
            %         response = iMatrix2Tensor(response);
            %     end
            % else
            %     response = iMatrix2Tensor(self.TableData{indices,2:end});
            % end
            
            output = self.outputFile_.(self.outputFieldName)(indices, :);
            response = iMatrix2Tensor(output);            
        end
                     
        function dataIndices = computeDataIndices(self)
            % computeDataIndices    Compute the indices into the data from
            % start and end index
            startIdx = min(self.StartIndexOfCurrentMiniBatch, self.NumberOfObservations);
            endIdx = startIdx + self.MiniBatchSize - 1;
            endIdx = min(endIdx, self.NumberOfObservations);
            
            dataIndices = startIdx:endIdx;
            
            % Convert sequential indices to ordered (possibly shuffled) indices
            dataIndices = self.OrderedIndices(dataIndices);
        end
        
        function advanceCurrentMiniBatchIndices(self)
            self.StartIndexOfCurrentMiniBatch = self.StartIndexOfCurrentMiniBatch + self.MiniBatchSize;        
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function inputStruct = parseInputs(self,varargin)
            
            narginchk(2,inf) % Use input parser to validate upper end of range.
            
            p = inputParser();
            
            p.addRequired('outputSize',@outputSizeValidator);
            % p.addRequired('X');
            % p.addOptional('Y',[]);
            p.addParameter('DataAugmentation','none',@augmentationValidator);
            
            colorPreprocessing = 'none';
            p.addParameter('ColorPreprocessing','none',@colorPreprocessingValidator);
            
            
            outputSizeMode = 'resize';
            p.addParameter('OutputSizeMode','resize',@outputSizeModeValidator);
            
            backgroundExecutionValidator = @(TF) validateattributes(TF,...
                {'numeric','logical'},{'scalar','real'},mfilename,'BackgroundExecution');
            p.addParameter('BackgroundExecution',false,backgroundExecutionValidator);
            
            % responseNames = [];
            % if istable(varargin{2})
            %     tbl = varargin{2};
            %     if (length(varargin) > 2) && (ischar(varargin{3}) || isstring(varargin{3}) || iscellstr(varargin{3}))
            %         if checkValidResponseNames(varargin{3},tbl)
            %             responseNames = varargin{3};
            %             varargin(3) = [];
            %         end
            %     end
            % end
            
            p.parse(varargin{:});
            inputStruct = p.Results;
            
            self.DataAugmentation = inputStruct.DataAugmentation;
            self.OutputSize = inputStruct.outputSize(1:2);
            self.OutputSizeMode = outputSizeMode;
            self.ColorPreprocessing = colorPreprocessing;
            % self.UseParallel = inputStruct.BackgroundExecution;
            
            % % Check if Y was specified for table or imageDatastore inputs.
            % propertiesWithDefaultValues = string(p.UsingDefaults);
            % if (isa(inputStruct.X,'matlab.io.datastore.ImageDatastore') || isa(inputStruct.X,'table')) && ~any(propertiesWithDefaultValues == "Y")
            %     error(message('nnet_cnn:augmentedImageSource:invalidYSpecification',class(inputStruct.X)));
            % end
            
            % if ~isempty(responseNames)
            %     inputStruct.X = selectResponsesFromTable(inputStruct.X,responseNames);
            %     inputStruct.Y = responseNames;
            % end
            
            % % Validate numeric inputs
            % if isnumeric(inputStruct.X)
            %     validateattributes(inputStruct.X,{'single','double','logical','uint8','int8','uint16','int16','uint32','int32'},...
            %         {'nonsparse','real'},mfilename,'X');
            % 
            %     validateattributes(inputStruct.Y,{'single','double','logical','uint8','int8','uint16','int16','uint32','int32','categorical'},...
            %         {'nonsparse','nonempty'},mfilename,'Y');
            % end
            
            % try
            %     self.DatasourceInternal = nnet.internal.cnn.MiniBatchDatasourceFactory.createMiniBatchDatasource(inputStruct.X,inputStruct.Y);
            % catch ME
            %     throwAsCaller(ME);
            % end
            
            function TF = colorPreprocessingValidator(sIn)
                colorPreprocessing = validatestring(sIn,{'none','rgb2gray','gray2rgb'},...
                    mfilename,'ColorPreprocessing');
                
                TF = true;
            end
            
            function TF = outputSizeModeValidator(sIn)
                outputSizeMode = validatestring(sIn,...
                    {'resize','centercrop','randcrop'},mfilename,'OutputSizeMode');
                
                TF = true;
            end
            
            function TF = outputSizeValidator(sizeIn)
                
                validateattributes(sizeIn,...
                    {'numeric'},{'vector','integer','finite','nonsparse','real','positive'},mfilename,'OutputSize');
                
                if (numel(sizeIn) ~= 2) && (numel(sizeIn) ~=3)
                    error(message('nnet_cnn:augmentedImageSource:invalidOutputSize'));
                end
                
                TF = true;
                
            end
            
        end
        
        function determineExpectedOutputSize(self)
            
            % If a user specifies a ColorPreprocessing option, we know the
            % number of channels to expect in each mini-batch. If they
            % don't specify a ColorPreprocessing option, we need to look at
            % an example from the underlying Datasource and assume all
            % images will have a consistent number of channels when forming
            % mini-batches.
            if strcmp(self.ColorPreprocessing,'rgb2gray')
                self.OutputRowsColsChannels = [self.OutputSize,1];
            elseif strcmp(self.ColorPreprocessing,'gray2rgb')
                self.OutputRowsColsChannels = [self.OutputSize,3];
            elseif strcmp(self.ColorPreprocessing,'none')
                % TODO:
                origMiniBatchSize = self.MiniBatchSize;
                self.DatasourceInternal.MiniBatchSize = 1;
                X = self.DatasourceInternal.nextBatch();
                self.DatasourceInternal.MiniBatchSize = origMiniBatchSize;
                self.DatasourceInternal.reset();
                exampleNumChannels = size(X,3);
                self.OutputRowsColsChannels = [self.OutputSize, exampleNumChannels];
            else
                assert(false,'Unexpected ColorPreprocessing option.');
            end
            
        end
        
        function Xout = applyAugmentationPipelineToBatch(self,X)
            if iscell(X)
                Xout = cellfun(@(c) self.applyAugmentationPipeline(c),X,'UniformOutput',false);
            else
                batchSize = size(X,4);
                Xout = zeros(self.OutputRowsColsChannels,'like',X);
                for obs = 1:batchSize
                    temp = self.preprocessColor(X(:,:,:,obs));
                    temp = self.augmentData(temp);
                    Xout(:,:,:,obs) = self.resizeData(temp);
                end
            end
        end    
        
        function Xout = applyAugmentationPipeline(self,X)
            if isequal(self.ColorPreprocessing,'none') && (size(X,3) ~= self.OutputRowsColsChannels(3))
               error(message('nnet_cnn:augmentedImageSource:mismatchedNumberOfChannels','''ColorPreprocessing'''));
            end
            temp = self.preprocessColor(X);
            temp = self.augmentData(temp);
            Xout = self.resizeData(temp);
        end
        
        function miniBatchData = augmentData(self,miniBatchData)
            if ~strcmp(self.DataAugmentation,'none')
                miniBatchData = self.ImageAugmenter.augment(miniBatchData);
            end
        end
        
        function Xout = resizeData(self,X)
            
            inputSize = size(X);
            if isequal(inputSize(1:2),self.OutputSize)
                Xout = X; % no-op if X is already desired Outputsize
                return
            end
            
            if strcmp(self.OutputSizeMode,'resize')
                Xout = augmentedImageSource.resizeImage(X,self.OutputSize);
            elseif strcmp(self.OutputSizeMode,'centercrop')
                Xout = augmentedImageSource.centerCrop(X,self.OutputSize);
            elseif strcmp(self.OutputSizeMode,'randcrop')
                Xout = augmentedImageSource.randCrop(X,self.OutputSize);
            end
        end
        
        function Xout = preprocessColor(self,X)
            
            if strcmp(self.ColorPreprocessing,'rgb2gray')
                Xout = convertRGBToGrayscale(X);
            elseif strcmp(self.ColorPreprocessing,'gray2rgb')
                Xout = convertGrayscaleToRGB(X);
            elseif strcmp(self.ColorPreprocessing,'none')
                Xout = X;
            end
        end
    end
end

function dataStore = iCreateDatastoreFromTable( aTable, shuffleIdx )

% Assume the first column of the table contains the paths to the images
if nargin < 2
    filePaths = aTable{:,1}'; % 1:end
else
    filePaths = aTable{shuffleIdx, 1}'; % Specific shuffle order
end

if any( cellfun(@isdir,filePaths) )
    % Directories are not valid paths
    iThrowWrongImagePathException();
end
try
    dataStore = imageDatastore( filePaths );
catch e
    iThrowFileNotFoundAsWrongImagePathException(e);
    iThrowInvalidStrAsEmptyPathException(e);
    rethrow(e)
end
% numObservations = size( aTable, 1 );
% numFiles = numel( dataStore.Files );
% if numFiles ~= numObservations
%     % If some files were discarded when the datastore was created, those
%     % files were not valid images and we should error out
%     iThrowWrongImagePathException();
% end
end

function iThrowWrongImagePathException()
% iThrowWrongImagePathException   Throw a wrong image path exception
exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatasource:WrongImagePath');
throwAsCaller(exception)
end

function iThrowFileNotFoundAsWrongImagePathException(e)
% iThrowWrongImagePathException   Throw a
% MATLAB:datastoreio:pathlookup:fileNotFound as a wrong image path
% exception.
if strcmp(e.identifier,'MATLAB:datastoreio:pathlookup:fileNotFound')
    iThrowWrongImagePathException()
end
end

function iThrowInvalidStrAsEmptyPathException(e)
% iThrowInvalidStrAsEmptyPathException   Throws a
% pathlookup:invalidStrOrCellStr exception as a EmptyImagePaths exception
if (strcmp(e.identifier,'MATLAB:datastoreio:pathlookup:invalidStrOrCellStr'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:internal:cnn:TableMiniBatchDatasource:EmptyImagePaths');
    throwAsCaller(exception)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(errorID, getString(message(errorID, varargin{:})));
end

function responseNames = iResponseNamesFromTable( tableData )
responseNames = tableData.Properties.VariableNames(2:end);
% To be consistent with ClassNames, return a column array
responseNames = responseNames';
end

function tensorResponses = iMatrix2Tensor( matrixResponses )
% iMatrix2Tensor   Convert a matrix of responses of size numObservations x
% numResponses to a tensor of size 1 x 1 x numResponses x numObservations
[numObservations, numResponses] = size( matrixResponses );
tensorResponses = matrixResponses';
tensorResponses = reshape(tensorResponses,[1 1 numResponses numObservations]);
end

function im = convertRGBToGrayscale(im)
if ndims(im) == 3
    im = rgb2gray(im);
end
end

function im = convertGrayscaleToRGB(im)
if size(im,3) == 1
    im = repmat(im,[1 1 3]);
end
end

function TF = augmentationValidator(valIn)

if ischar(valIn) || isstring(valIn)
    TF = string('none').contains(lower(valIn)); %#ok<STRQUOT>
elseif isa(valIn,'imageDataAugmenter') && isscalar(valIn)
    TF = true;
else
    TF = false;
end

end
