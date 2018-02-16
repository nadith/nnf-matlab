classdef SHMDatGenerator < handle
    % SHMDATAGENERATOR represents the base class for all SHM Models data generators.
    %   Extend this class and override `generate()` method to build custom model specific data
    %   generator.
   
    % Copyright 2015-2018 Nadith Pathirage, Curtin University (chathurdara@gmail.com). 
    properties (SetAccess = public)
        name;           % (s) Name of the object; Use it as a prefix when writing data to the disk.
        mode;           % Operating mode for the `SHMDataGenerator`.
        damage_cases;  	% Each cell indicating the element indices to introduce damage. Magnitude of the 
                        % damage is speicified via `stiff_range`.
        stiff_range;    % Damage magnitudes. (1 -> intact, 0-> fully damaged).
    end
    
    properties (SetAccess = protected)
        params_;
    end
    
    methods (Access = protected, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [samples, ulbl, llbl] = generate_single_element_cases_(ele_count, stiff_values)
            % GENERATE_SINGLE_ELEMENT_CASES: Generate samples matrix for single element damage cases.
            %
            % Parameters
            % ----------
            % ele_count : int
            %       Total elemental stiffness parameters count. This is also used as the index of
            %       the elements to introduce damages in varying magnitude specified in `stiff_values`.
            %
            % stiff_values : vector -double
            %       Damage magnitudes (stiffness reductions) per each element.
            %
            % Returns
            % -------
            % samples : 2D matrix -double
            %       Samples that correspond to varying magnitude at different location damages.
            %       Single elements damage scnarios are only considered.
            %
            % ulbl : vector -uint64
            %       Unique label for each damage pattern defined for elements denoted by `ele_count'.
            %       i.e [0.5; 0; 0; ...], [0.6; 0; 0; ...] will be assigned two unique labels.
            %
            % llbl : vector -uint64
            %       Location label for damage patterns of varying magnitude that corresponds to
            %       same location.
            %       i.e [0.5; 0; 0; ...], [0.6; 0; 0; ...] will be assigned lbl=1.
            %           [0; 0.5; 0; ...], [0; 0.6; 0; ...] will be assigned lbl=2.
            %
            
            n_stiff_values = numel(stiff_values);
            samples = ones(ele_count, n_stiff_values * ele_count);
            
            % Unique label for location and magnitude of a damage pattern
            ulbl = zeros(1, n_stiff_values*ele_count);
            
            % Local label for the location of the damage
            llbl = zeros(1, n_stiff_values*ele_count);
            
            st = 1;
            for i=1:ele_count
                en = st + numel(stiff_values)-1;
                samples(i, st:en) = stiff_values;
                ulbl(st:en) = st:en;
                llbl(st:en) = repmat(i, 1, numel(stiff_values));
                st = en + 1;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [samples, ulbl, llbl] = generate_multi_element_cases_(ele_count, stiff_values, ele_indices)
            % GENERATE_MULTI_ELEMENT_CASES: Generate samples matrix for multiple element damage cases.
            %
            % Parameters
            % ----------
            % ele_count : int
            %       Total elemental stiffness parameters count.
            %
            % stiff_values : vector -double
            %       Damage magnitudes (stiffness reductions) per each element index specified
            %       in `ele_indices`.
            %
            % ele_indices : int
            %       Elemental stiffness parameter index to introduce damages in varying magnitude
            %       specified in `stiff_values`.
            %
            % Returns
            % -------
            % samples : 2D matrix -double
            %       Samples that correspond to varying magnitude at different location damages.
            %       Multiple elements damage scnarios (as specified in `ele_indices`) are considered.
            %
            % ulbl : vector -uint64
            %       Unique label for each damage pattern defined for elements denoted by `ele_count'.
            %       i.e [0.5; 0.5; 0; ...], [0.5; 0.6; 0; ...] will be assigned two unique labels.
            %
            % llbl : vector -uint64
            %       Location label for damage patterns of varying magnitude that corresponds to
            %       same location.
            %       i.e [0.5; 0.5; 0; ...], [0.5; 0.6; 0; ...] will be assigned lbl=1.
            %
            
            n_stiff_values = numel(stiff_values);
            n_ele_indices = numel(ele_indices);
            samples = ones(ele_count, (n_stiff_values^n_ele_indices));
            
            % Unique label for location and magnitude of a damage pattern
            ulbl = 1:n_stiff_values^n_ele_indices;
            
            % Local label for the location of the damage
            llbl = ones(1, n_stiff_values^n_ele_indices);
            
            for i=1:n_ele_indices
                ele_idx = ele_indices(i);
                ss = repmat(stiff_values', 1, n_stiff_values^(n_ele_indices-i))';
                samples(ele_idx, :) = repmat(ss(:)', 1, n_stiff_values^(i-1));
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = SHMDatGenerator(name, mode, damage_cases, stiff_range, varargin)
            % Constructs a `SHMDatGenerator` object.
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
            
            % Imports
            import civil.shm.generators.Mode;            
            
            disp(['Costructor::SHMDatGenerator ' name]);
            self.name = name;            
            if (isempty(mode)); mode = Mode.MEMORY_MODE; end
            self.mode = mode;
            self.damage_cases = damage_cases;
            self.stiff_range = stiff_range;
            
            p = inputParser;
            
            defaultDataSaveDirectory = [];
            defaultUncertainityLevel = 0;
            defaultUncertainityPerSample = [];
            defaultMeasurementNoisePerSample = [];
            defaultMeasurementNoiseOnFreq = 0;
            defaultMeasurementNoiseOnModeShape = 0;
            GenerateForElementModel21 = false;
            
            addParameter(p, 'DataSaveDirectory', defaultDataSaveDirectory);
            addParameter(p, 'UncertainityLevel', defaultUncertainityLevel);
            addParameter(p, 'UncertainityPerSample', defaultUncertainityPerSample);
            addParameter(p, 'MeasurementNoisePerSample', defaultMeasurementNoisePerSample);
            addParameter(p, 'MeasurementNoiseOnFreq', defaultMeasurementNoiseOnFreq);
            addParameter(p, 'MeasurementNoiseOnModeShape', defaultMeasurementNoiseOnModeShape);
            addParameter(p, 'GenerateForElementModel21', GenerateForElementModel21);
            
            parse(p, varargin{:});
            self.params_ = p.Results;
            
            % Error handling
            if (mode == Mode.DISK_MODE || mode == Mode.MEMORY_DISK_MODE)
                if isempty(self.params_.DataSaveDirectory)
                    error(['DISK_MODE: Please specify location to save in the paramter `DataSaveDirectory`']);
                end
            end
            
            if isempty(damage_cases)
                error(['damage_cases are undefined']);
            end            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Abstract, Access = public)
        [data] = generate(self);
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Samples, Output_ori, ulbl, llbl] = ...
                                    generate_uncrt_samples_(self, intact_Sample, Samples, ulbl, llbl)
            % Adding uncertainties
            
            p = self.params_;
            Output_ori = [];
            
            if ~isempty(p.UncertainityPerSample)
                Samples = reshape(repmat(Samples', 1, p.UncertainityPerSample)', size(Samples, 1), []);
                Output_ori = bsxfun(@minus, intact_Sample, Samples);
                
                % Add uncertainities to the samples matrix; hence this is used in
                % `Gen_FM_DeepLearning_basic` script
                Uncertainty = 1 + p.UncertainityLevel * randn(size(Samples));
                Samples = Samples .* Uncertainty;
                
                % Update labels
                ulbl = reshape(repmat(ulbl', 1, p.UncertainityPerSample)', size(ulbl, 1), []);
                llbl = reshape(repmat(llbl', 1, p.UncertainityPerSample)', size(llbl, 1), []);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ModalInfo, ModalInfo_Noise, Output, ulbl, llbl, Output_ori, Samples] = ...
                add_measurement_noise_(self, ModalInfo, Output, ulbl, llbl, Output_ori, Samples)
            % Adding measurement noise
            
            p = self.params_;
            ModalInfo_Noise = [];
            
            if ~isempty(p.MeasurementNoisePerSample)
                
                % std across all samples
                std_mi = std(ModalInfo, [], 2);
                
                ModalInfo = reshape(repmat(ModalInfo', 1, p.MeasurementNoisePerSample)', size(ModalInfo, 1), []);
                Output = reshape(repmat(Output', 1, p.MeasurementNoisePerSample)', size(Output, 1), []);
                
                if ~isempty(Output_ori)
                    Output_ori = reshape(repmat(Output_ori', 1, p.MeasurementNoisePerSample)', size(Output_ori, 1), []);
                end
                
                if ~isempty(Samples)
                    Samples = reshape(repmat(Samples', 1, p.MeasurementNoisePerSample)', size(Samples, 1), []);
                end
                
                % Update labels
                ulbl = reshape(repmat(ulbl', 1, p.MeasurementNoisePerSample)', size(ulbl, 1), []);
                llbl = reshape(repmat(llbl', 1, p.MeasurementNoisePerSample)', size(llbl, 1), []);
                                
                % Generate noise from std gaussian distribution (mean=0, std=1)
                % Hence (std_mi .* noise) => (mean=0, std=std_mi) gauassian
                noise = randn(size(ModalInfo, 1), size(ModalInfo, 2));

                % Frequency Noise (7 frequencies)
                ModalInfo_Noise(1:7, :) = ModalInfo(1:7, :) + p.MeasurementNoiseOnFreq * std_mi(1:7) .* noise(1:7, :);
                
                % Mode Shape Noise (7*14 mode shapes)
                ModalInfo_Noise(8:105, :) = ModalInfo(8:105, :) + p.MeasurementNoiseOnModeShape * std_mi(8:105) .* noise(8:105, :);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

