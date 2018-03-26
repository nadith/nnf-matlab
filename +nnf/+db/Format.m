classdef Format
    %FORMAT Enumeration describes the format of the NNdb database.
    enumeration      
        H_W_CH_N,     % =0 Height x Width x Channels x Samples (image db_format) 
        H_N,          % Height x Samples
        N_H_W_CH,     % Samples x Height x Width x Channels (image db_format) 
        N_H,          % Samples x Height
    end
end