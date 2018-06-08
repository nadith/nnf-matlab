% DO NOT run startup.m in deployed mode 
if isdeployed
    return
end

% Root folder
base_path = ['E:/Clouds/Google Drive/Curtin Univeristy (work)/Matlab/src'];

% Workspace folder
w_folder = 'ml';

% Change the current directory to root folder
eval(['cd ' '''' base_path '''']);

% Add path to matlab: env_init, env_load_im_db must be at this path
addpath (base_path); 

% Add path to matlab (recursive): current workspace 
global w_path;
w_path = [base_path '/' w_folder];

 % Add the current working directory to path
addpath (w_path);
infos = dir(w_path);
for i=1:numel(infos)                
    name = infos(i).name;
    % Exclude these items in the current iteraing directory
    if (strcmp(name, '.') || strcmp(name, '..') || ...
        name(1) == '.' || strcmp(name, '_other') || strcmp(name, '+nnf') || strcmp(name, '+civil'))
        continue;
    end
    
    % Add only directories to the path
    if (infos(i).isdir)
        addpath(genpath ([w_path '/' name]));
    end
end

% Add path to matlab (recursive): selective external libs
%wlib_folder = 'DeepLearnToolbox';
%wlib_path = [base_path '/ext_libs/' wlib_folder];
%addpath (genpath (wlib_path));

%wlib_folder = 'INface_tool';
%wlib_path = [base_path '/ext_libs/' wlib_folder];
%addpath (genpath (wlib_path));

%wlib_folder = 'Deng Cai';
%wlib_path = [base_path '/ext_libs/' wlib_folder];
%addpath (genpath (wlib_path));

%wlib_folder = 'efficientLBP';
%wlib_path = [base_path '/ext_libs/' wlib_folder];
%addpath (genpath (wlib_path));

%wlib_folder = 'drtoolbox';
%wlib_path = [base_path '/ext_libs/' wlib_folder];
%addpath (genpath (wlib_path));

% Change director to the working folder
eval(['cd ' w_folder]);

env_init();

% Loading databases
imdb_path = ['E:/Clouds/Google Drive/Curtin Univeristy (work)/Matlab/data'];

% For Linux
% script_path = '~/MatlabWorkspace/Scripts';
% addpath (genpath ([script_path]));
% imdb_path = '~/MatlabWorkspace/Data/';
% cd(script_path);

%% Cloud databases
% Multi Pie Database
env_load_im_db('imdb_mp', [imdb_path '/IMDB_66_66_MP_33.mat']);

%% Local databases
imdb_path = 'F:/#Research Data/FaceDB';

% ORL Database
env_load_im_db('imdb_orl', [imdb_path '/IMDB_64_64_ORL_8.mat']);

% YaleB
env_load_im_db('imdb_yaleb', [imdb_path '/IMDB_66_58_YALEB_64.mat']);

% AR Database
env_load_im_db('imdb_ar', [imdb_path '/IMDB_66_66_AR_12.mat']);

% Curtin Database
%env_load_im_db('imdb_curtin', [imdb_path '/IMDB_66_66_CUR_8.mat']);

% Both AR and CURTIN
%env_load_im_db('imdb_ar_cur', [imdb_path '/IMDB_66_66_AR_CUR_8.mat']);

% LFW Database
env_load_im_db('imdb_lfw', [imdb_path '/IMDB_64_64_LFW_SLINDA_9.mat']);


% Clear unnecessary variables
%clear wlib_path;
clear i;
clear infos;
clear name;
clear base_path;
clear w_folder;
clear w_path;
clear imdb_path;