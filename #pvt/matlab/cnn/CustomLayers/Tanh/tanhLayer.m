function layer = tanhLayer(varargin)
% tanhLayer   Rectified linear unit (Tanh) layer
%
%   layer = tanhLayer() creates a rectified linear unit layer. This type of
%   layer performs a simple threshold operation, where any input value
%   less than zero will be set to zero.
%
%   layer = tanhLayer('PARAM1', VAL1) specifies optional parameter
%   name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       Create a rectified linear unit layer.
%
%       layer = tanhLayer();
%
%   See also leakyReluLayer, clippedReluLayer, nnet.cnn.layer.tanhLayer,
%   convolution2dLayer, fullyConnectedLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a Tanh layer.
internalLayer = Tanh(inputArguments.Name);

% Pass the internal layer to a  function to construct a user visible Tanh
% layer.
layer = TanhLayerCls(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;
defaultName = '';
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end
