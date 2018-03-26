function data = mwpp_load_shm_data(filename)
    % MWPP_LOAD_SHM_DATA: Read the custom format mining data file and return the `data` struct with 
    %       mandatory fields.
    %
    % Parameters
    % ----------
    % filename : string
    %       Path to the data file.
    %            
    % Returns
    % -------
    % data : struct
    %       Containing the following fields;
    %           Input: Input data - Mandatory
    %               2D tensor: In the format = H X N
    %
    %           Output: Input data matrix (Format: H X N) - Mandatory      
    %               []: Empty for classification problems
    %               2D tensor: In the format = H X N for regression problems
    %
    %           ulbl: Unique label vector - Mandatory        
    %               Used to handle the class boundary in `MovingWindowPreProcessor`.
    %               1D row vector: In the format 1 x N
    
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com). 
    
    disp(['Processing: ' filename]);
    mat = load(filename);

    % This will provide smooth image patterns for the SHM data
    import nnf.utl.hpfilter_2017a;
    hpf = hpfilter_2017a;    
    data.Input = zscore(filter(hpf, mat.ModalInfo'))';
    data.Output = single(mat.Output_ori);
    data.ulbl = mat.ulbl;
    
    % data.Input = zscore(mat.ModalInfo')'; 
    % data.Output = single(mat.Output_ori);
    % data.ulbl = mat.ulbl;
        
    % data.Input = mat.ModalInfo;
    % data.Output = single(mat.Output_ori);
    % data.ulbl = mat.ulbl;
end