classdef Mode < uint32
    % Mode Enumeration describes the operating mode for the `SHMDataGenerator`.
    enumeration      
        MEMORY_MODE (1)     % Data generation happens in memory and all data generated will be
                            % returned to the caller.
        DISK_MODE (2)       % Data generation happens in memory in parts and written to the disk. 
                            % No data is returned to the caller. 
        MEMORY_DISK_MODE (3)% Data generation happens in memory in parts and written to the disk. 
                            % All generated data will be returned to the caller. 
    end
end