function env_load_im_db( db_var_name, db_path )
%env_load_im_db Summary of this function goes here
%   Detailed explanation goes here
    if (exist(db_path, 'file') == 2) % File exists        
        load_struct = load(db_path); 
        assignin('base', db_var_name, load_struct.imdb_obj.db);               
        assignin('base', sprintf('%s_class', db_var_name), load_struct.imdb_obj.class);                
        assignin('base', sprintf('%s_im_per_class', db_var_name), load_struct.imdb_obj.im_per_class);         
        evalin('base', sprintf('disp([''%s size -> '' num2str(size(%s))])', db_var_name, db_var_name));             
    end
end

