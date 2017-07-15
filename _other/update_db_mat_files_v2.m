% Update the mat database files to have all fields properly set.

% Set the path to the directory containing all database mat files.
dir_db = 'D:\matdbs';

dir_infos = dir(dir_db);
dir_infos = dir_infos(~ismember({dir_infos.name},{'.','..'})); % exclude '.' and '..'

% Sort the folder names (class names)
[~,ndx] = natsortfiles({dir_infos.name});   % indices of correct order
dir_infos = dir_infos(ndx);                 % sort structure using indices

import nnf.db.NNdb;

% Iterator
for i = 1:length(dir_infos)
    filename = dir_infos(i).name;    
    disp(['Processing database file: ' filename]);
    
    loaded = load(fullfile(dir_db, filename));
    nndb = NNdb('TEMP', loaded.imdb_obj.db, double(loaded.imdb_obj.im_per_class), true);
    nndb.save_compressed(fullfile(dir_db, filename));
end