classdef Format
    %NNDBFORMAT Enumeration describes the format of the NNdb database.    
    enumeration      
      H_W_CH_N,     % =0 Height x Width x Channels x Samples (image db format) 
      H_W_CH_N_NP,  % Height x Width x Channels x Samples x PatchCount
      H_N,          % Height x Samples
      H_N_NP        % Height x Samples x PatchCount
    end
end
        

