classdef SHM7SDatGenerator < civil.shm.generators.SHMDatGenerator
    % SHM7SDATAGENERATOR generates data for 7 storey bulding.
    %   The 7-Storey SHM model is as shown below:    
    %
    % Number denote the elemental stiffness parameter index.
    %       ___  __  __  ___
    %  21 |  46  47  48  49   | 50 
    %  20 |                   | 51
    %  19 |                   | 52
    %       ___  __  __  ___
    %  18 |  42  43  44  45   | 53 
    %  17 |                   | 54
    %  16 |                   | 55
    %       ___  __  __  ___
    %  15 |  38  39  40  41   | 56 
    %  14 |                   | 57
    %  13 |                   | 58
    %       ___  __  __  ___
    %  12 |  34  35  36  37   | 59 
    %  11 |                   | 60
    %  10 |                   | 61
    %       ___  __  __  ___
    %  09 |  30  31  32  33   | 62 
    %  08 |                   | 63
    %  07 |                   | 64
    %       ___  __  __  ___
    %  06 |  26  27  28  29   | 65 
    %  05 |                   | 66
    %  04 |                   | 67
    %       ___  __  __  ___
    %  03 |  22  23  24  25   | 68 
    %  02 |                   | 69
    %  01 |                   | 70
        
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com).    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
        function combine_files(filepaths, save_filepath)
            %
            % Util code to combine files generate via `SHM7SDatGenerator` in Mode.DISK_MODE
            % into a one file.

            ModalInfo = [];
            ModalInfo_Noise = [];
            Output = [];
            Output_ori = [];
            ulbl = [];
            llbl = [];
            
            for i=1:numel(filepaths)
                file = filepaths{i};
                f = load(file);        
                
                ModalInfo = [ModalInfo f.ModalInfo];
                ModalInfo_Noise = [ModalInfo_Noise f.ModalInfo_Noise];
                Output = [Output f.Output];
                Output_ori = [Output_ori f.Output_ori];
                ulbl = [ulbl f.ulbl];
                llbl = [llbl f.llbl];
            end    
            f.ModalInfo = ModalInfo;
            f.ModalInfo_Noise = ModalInfo_Noise;
            f.Output = Output;
            f.Output_ori = Output_ori;
            f.ulbl = ulbl;
            f.llbl = llbl;
            
            % Preprocessing
            hpf = hpfilter_2017a;
            f.ModalInfo = filter(hpf, f.ModalInfo')';
            f.ModalInfo_Noise = filter(hpf, f.ModalInfo_Noise')'; 
            
            save(save_filepath, '-struct', 'f');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ModalInfo, Output] = RunDataGenScript(intact_Sample, Samples)
                % Make sure local variables `intact_Sample` `Samples` are already set before invoking
                % the following script
                gen_21_element = false;
                Gen_FM_DeepLearning_basic;
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
        
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        function self = SHM7SDatGenerator(name, mode, damage_cases, stiff_range, varargin) 
            % Constructs a `SHM7SDatGenerator` object.
            %
            % Parameters
            % ----------
            % name : string
            %       Name of the `SHMDatGenerator` object.
            %
            % mode : Enumeration `civil.shm.generators.Mode`
            %       Operating mode for the `SHMDataGenerator`. (Default value = Mode.MEMORY_MODE).
            %
            % damage_cases : cell array
            %       Each cell indicating the element indices to introduce damage. Magnitude of the 
            %       damage is speicified via `stiff_range`.
            %
            % stiff_range : vector -double
            %       Damage magnitudes. (1 -> intact, 0-> fully damaged).
            %
            % varargin : 
            %       DataSaveDirectory : string
            %       UncertainityLevel : double
            %       UncertainityPerSample : int
            %       MeasurementNoisePerSample : int
            %       MeasurementNoiseOnFreq : double
            %       MeasurementNoiseOnModeShape : double
            %       GenerateForElementModel21 : bool
            %
            
            self = self@civil.shm.generators.SHMDatGenerator(name, mode, damage_cases, stiff_range, varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        function data = generate_all_se(self)
            % GENERATE_ALL_SE: Generate input (ModalInfo) and Output for all single element damage
            % cases.
            %
            %   TODO: This method does not support writing to the disk
            %
            % Returns
            % -------
            % data : struct
            %       ModalInfo, Output, Samples, [ModalInfo_Noise], [Output_ori], ulbl, llbl, etc... 
            %       fields will be returned.
            %       - `Output_ori` will not be empty only when uncertainities are considered in 
            %           generating data.
            %       - 'ModalInfo_Noise' will not be empty only when measurement noise is considered in 
            %           generating data. 
            %
            
            % Imports
            import civil.shm.generators.SHMDatGenerator;
            
            p = self.params_;           
            
            % Set this variable for `Gen_FM_DeepLearning_basic` to generate the `ModalInfo` and `Output`
            intact_Sample = ones(70, 1);            
            
            % Samples: (1-> 100% damage, 0 -> 0% damage)
            [Samples, ulbl, llbl]= SHMDatGenerator.generate_single_element_cases_(70, self.stiff_range);
                     
            if (~p.GenerateForElementModel21)
                % TODO: Implement
                % Make sure `samples` are compatible for 21 element model
            end
            
            % Add INTACT state
            Samples = [intact_Sample Samples];
            ulbl = [0 ulbl];  % ulbl for intact state
            llbl = [0 llbl];  % llbl for intact state
            
            % Adding uncertanities
            [Samples, data.Output_ori, ulbl, llbl] = self.generate_uncrt_samples_(intact_Sample, Samples, ulbl, llbl);   
                                                
            % Make sure the variable `Samples` is set before invoking this script
            gen_21_element = false;
            Gen_FM_DeepLearning_basic;
            
            % Adding Measurement noise
            [ModalInfo, data.ModalInfo_Noise, Output, ulbl, llbl, data.Output_ori, Samples] = ...
                self.add_measurement_noise_(ModalInfo, Output, ulbl, llbl, data.Output_ori, Samples);  
                                        
            % Update the fields
            data.ModalInfo = ModalInfo;
            data.Output = Output;
            data.Samples = Samples;            
            data.ulbl = ulbl;
            data.llbl = llbl;
            data.Model21 = p.GenerateForElementModel21;
            
            % Update uncertainity related fields
            data.UncertainityPerSample = p.UncertainityPerSample;
            data.UncertainityLevel = p.UncertainityLevel;

            % Update measurement noise related fields
            data.MeasurementNoisePerSample = p.MeasurementNoisePerSample;
            data.MeasurementNoiseOnFreq = p.MeasurementNoiseOnFreq;
            data.MeasurementNoiseOnModeShape = p.MeasurementNoiseOnModeShape;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data = generate(self)
            % GENERATE: Generate input (ModalInfo) and Output for specified `damage_cases`.
            %
            % Returns
            % -------
            % data : struct
            %       ModalInfo, Output, Samples, [ModalInfo_Noise], [Output_ori], ulbl, llbl, etc... 
            %       fields will be returned.
            %       - `Output_ori` will not be empty only when uncertainities are considered in 
            %           generating data.
            %       - 'ModalInfo_Noise' will not be empty only when measurement noise is considered in 
            %           generating data. 
            %
            
            % Imports
            import civil.shm.generators.Mode;
            import civil.shm.generators.SHMDatGenerator;
            import civil.shm.generators.SHM7SDatGenerator;
            
            p = self.params_;

            % Initialize the output
            data.ModalInfo = [];            
            data.Output = [];
            data.Samples = [];
            data.ModalInfo_Noise = [];  % Populated only if measurement noise is added
            data.Output_ori = [];       % Populated only if uncertainities are added
            data.ulbl = [];
            data.llbl = [];
            data.Model21 = p.GenerateForElementModel21;
            
            % Update uncertainity related fields
            data.UncertainityPerSample = p.UncertainityPerSample;
            data.UncertainityLevel = p.UncertainityLevel;

            % Update measurement noise related fields
            data.MeasurementNoisePerSample = p.MeasurementNoisePerSample;
            data.MeasurementNoiseOnFreq = p.MeasurementNoiseOnFreq;
            data.MeasurementNoiseOnModeShape = p.MeasurementNoiseOnModeShape;
            
            % Set this variable for `Gen_FM_DeepLearning_basic` to generate the `ModalInfo` and `Output`
            intact_Sample = ones(70, 1);
                        
            ulbl = [0]; % ulbl for intact state
            llbl = [0]; % llbl for intact state           
                        
            ulbl_offset = ulbl(end);
            llbl_offset = llbl(end);           
            
            % Add INTACT state
            if (self.mode == Mode.MEMORY_MODE)
                data.Samples = [intact_Sample];
            
            elseif (self.mode == Mode.DISK_MODE || self.mode == Mode.MEMORY_DISK_MODE)
                [matObj, tmp_samples] = self.gen_and_write__(...
                        fullfile(p.DataSaveDirectory, [self.name '_' num2str(ulbl_offset) '.mat']), ...
                        intact_Sample, intact_Sample, ulbl, llbl);
                    
                if (self.mode == Mode.MEMORY_DISK_MODE)
                    data.ModalInfo = [data.ModalInfo matObj.ModalInfo];
                    data.Output = [data.Output matObj.Output];
                    data.Samples = [data.Samples tmp_samples];
                    data.ModalInfo_Noise = [data.ModalInfo_Noise matObj.ModalInfo_Noise];
                    data.Output_ori = [data.Output_ori matObj.Output_ori];
                    data.ulbl = [data.ulbl matObj.ulbl];
                    data.llbl = [data.llbl matObj.llbl];                       
                end
            end            
                   
            % Iterate throuh the combinations
            for dci=1:size(self.damage_cases, 1)
                
                % Fetch dci^th damage case
                damge_case = self.damage_cases{dci};
                
                % tmp_samples: (1-> 100% damage, 0 -> 0% damage)
                [tmp_samples, tmp_ulbl, tmp_llbl]= SHMDatGenerator.generate_multi_element_cases_(70, self.stiff_range, damge_case);
            
                % Correction, since tmp_ulbl, tmp_llbl starts from the begining everytime
                tmp_ulbl = ulbl_offset + tmp_ulbl;
                tmp_llbl = llbl_offset + tmp_llbl;
                
                % Append the new labels
                ulbl = [ulbl tmp_ulbl];
                llbl = [llbl tmp_llbl];

                % Offset for next iteration
                ulbl_offset = ulbl(end);
                llbl_offset = llbl(end);
                
                if (self.mode == Mode.MEMORY_MODE)
                    data.Samples = [data.Samples tmp_samples];
                    
                elseif (self.mode == Mode.DISK_MODE || self.mode == Mode.MEMORY_DISK_MODE)
                    [matObj, tmp_samples] = self.gen_and_write__(...
                        fullfile(p.DataSaveDirectory, [self.name '_' num2str(ulbl_offset) '.mat']), ...
                        intact_Sample, tmp_samples, tmp_ulbl, tmp_llbl);
                    
                    if (self.mode == Mode.MEMORY_DISK_MODE)
                        data.ModalInfo = [data.ModalInfo matObj.ModalInfo];
                        data.Output = [data.Output matObj.Output];
                        data.Samples = [data.Samples tmp_samples];
                        data.ModalInfo_Noise = [data.ModalInfo_Noise matObj.ModalInfo_Noise];
                        data.Output_ori = [data.Output_ori matObj.Output_ori];
                        data.ulbl = [data.ulbl matObj.ulbl];
                        data.llbl = [data.llbl matObj.llbl];
                    end
                end
            end
                        
            if (self.mode == Mode.MEMORY_MODE)
                % Set this variable for `Gen_FM_DeepLearning_basic` to generate the `ModalInfo` 
                % and `Output`
                Samples = data.Samples;                                
                
                % Adding uncertanities
                [Samples, data.Output_ori, ulbl, llbl] = self.generate_uncrt_samples_(intact_Sample, Samples, ulbl, llbl);            
                
                % Run the script
                [ModalInfo, Output] = SHM7SDatGenerator.RunDataGenScript(intact_Sample, Samples);
                                         
                % Adding Measurement noise
                [ModalInfo, data.ModalInfo_Noise, Output, ulbl, llbl, data.Output_ori, Samples] = ...
                    self.add_measurement_noise_(ModalInfo, Output, ulbl, llbl, data.Output_ori, Samples);  
                    
                % Update the fields
                data.ModalInfo = ModalInfo;
                data.Output = Output;
                data.Samples = Samples;            
                data.ulbl = ulbl;
                data.llbl = llbl;
            end    
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end   
    
    methods (Access = private)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        function [matObject, Samples] = gen_and_write__(self, filepath, intact_Sample, Samples, ulbl, llbl)
            % Invoked in disk mode
            
            % Imports
            import civil.shm.generators.Mode;
            assert(self.mode == Mode.DISK_MODE || self.mode == Mode.MEMORY_DISK_MODE);
            
            p = self.params_;            
            
            % Create the directory if does not exist
            [parentdir,~,~] = fileparts(filepath);
            [~, ~] = mkdir(parentdir);
            
            % Create a writable mat file to save data
            matObject = matfile(filepath, 'Writable', true);            
            
            % Adding uncertanities
            [Samples, matObject.Output_ori, matObject.ulbl, matObject.llbl] = ...
                self.generate_uncrt_samples_(intact_Sample, Samples, ulbl, llbl);
                        
            % Run the script
            [ModalInfo, Output] = SHM7SDatGenerator.RunDataGenScript(intact_Sample, Samples);
            
            % Adding Measurement noise
            [matObject.ModalInfo, ... 
                matObject.ModalInfo_Noise, ...
                matObject.Output, ...
                matObject.ulbl, matObject.llbl, ...
                matObject.Output_ori, ...
                Samples] = ...
                self.add_measurement_noise_(ModalInfo, Output, matObject.ulbl, matObject.llbl, matObject.Output_ori, Samples);            

            % matObject.Samples = Samples; % To reduce the size of the file            
            matObject.Model21 = p.GenerateForElementModel21;
            
            % Update uncertainity related fields
            matObject.UncertainityPerSample = p.UncertainityPerSample;
            matObject.UncertainityLevel = p.UncertainityLevel;

            % Update measurement noise related fields
            matObject.MeasurementNoisePerSample = p.MeasurementNoisePerSample;
            matObject.MeasurementNoiseOnFreq = p.MeasurementNoiseOnFreq;
            matObject.MeasurementNoiseOnModeShape = p.MeasurementNoiseOnModeShape;
        end
     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % [UNUSED] Discuss with JunLi and remove
        function dat = map_the_sign_to_health_case__(dat)
            % TODO: [UNUSED] Discuss with JunLi and remove
            
            location=[];
            health=[];
            [rows,cols] = size(dat);
            for i =7:76:rows
                [ma loc]=max(abs(dat(i:(i+76-1),1))); %find the maximum value and location of the healthy case for each modeshape
                location=[location i+loc-1]; %store the location
                health=[health sign(dat(i+loc-1))];%store the sign of maximum value
            end
            for i=2:cols % scan each mode
                for j =1:6 %6 mode shape
                    if (sign(dat(location(j),i))~=health(j)) % if the sign of the maximum value different
                        dat((7+(j-1)*76):(7+j*76-1),i)=-dat((7+(j-1)*76):(7+j*76-1),i); %change all the sign of this modeshape
                    end
                end
            end
            
            %  location=[];
            %  health=[];
            %  [rows,cols] = size(dat);
            %  for i =7:76:rows
            %      [ma loc]=max(dat(i:(i+76-1),1)); %find the maximum value and location of the healthy case for each modeshape
            %      location=[location loc]; %store the location
            %      health=[health sign(ma)];%store the sign of maximum value
            %  end
            %  for i=2:cols % scan each mode
            %      for j =1:6 %6 mode shape
            %          if (sign(dat(location(j),i))~=health(j)) % if the sign of the maximum value different
            %              dat((7+(j-1)*76):(7+j*76-1),i)=-dat((7+(j-1)*76):(7+j*76-1),i); %change all the sign of this modeshape
            %          end
            %      end
            %  end
            % for i = 1:rows
            %     health = sign(dat(i,1));
            %     for j = 1:cols
            %         if (sign(dat(i,j))~=health)
            %             dat(i,j) = - dat(i,j);
            %         end
            %     end
            % end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    end

end

