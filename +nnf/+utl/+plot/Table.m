classdef Table < handle
    % TABLE represents table structure for NNFramwork.
    %   Utilized in plot sub-package.
    %
    % Supported Table Struture
    % -------------------------
    % Method 1: 90 99 100
    % Method 2: xx xx xx
    % Method 3: xx xx xx
    %
    
    properties
        approaches;
        values;
        colors;
        line_widths;
        line_specs;
        y_axis_range;
        
        caption;
        label;
        title;
    end
    
    methods (Access = public, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function tab = get_example()
            % GET_EXAMPLE: Get an example table object.
            %
            % Notes
            % -----
            % i.e. Table Object Structure
            % Test 1: 10 20 30
            % Test 2: 40 50 60
            %
            
            % Imports
            import nnf.utl.plot.Table
            
            tab = Table('latex-caption', 'latex-label', 'fig-title');
            tab.approaches = {'Test1', 'Test2'};
            tab.values = [10 20 30; 40 50 60];
            tab.colors = {'red', 'blue'};
            tab.line_widths = [1, 1];
            tab.line_specs = {'-b*', '-b*'};
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function tab = get_example2()
            % GET_EXAMPLE2: Get an example table object.
            %
            % Notes
            % -----
            % i.e. Table Object Structure
            % SFDAE-SRC:  [100 100 100 100 97.3 98.7]
            % SFDAE-LDA:  [100 100 100 100 100 100]
            % SFDAE-PCA:  [98.7 98.7 94.7 97.3 94.7 97.3]
            % SRC:        [100 98.7 98.7 97.3 96 97.3]
            % LDA:        [100 100 100 98.7 97.3 100]
            % PCA:        [94.7 93.3 88 92 84 94.7]
            % SRC(R):     [98.7 98.7 100 100 97.3 98.7]
            % LDA(R):     [93.3 97.3 98.7 92 94.7 97.3]
            % PCA(R):     [60 80 86.7 96 89.3 90.7]
            % HSV-V(R):   [100 100 89.3 94.7 84 92]
            %
            
            % Imports
            import nnf.utl.plot.PlotMan
            import nnf.utl.plot.Table;
            
            tab = Table('Results of the tests performed with 75 identities in AR database.', 'tab:chap3:SDB', '');
            tab.approaches = {'HSV-V (Reconst)', 'PCA (Reconst)', 'LDA (Reconst)', 'SRC (Reconst)', 'PCA', 'LDA', 'SRC', 'AutoNet-PCA', 'AutoNet-LDA', 'AutoNet-SRC'};
            tab.values = [...
                [100 100 89.3 94.7 84 92]; [60 80 86.7 96 89.3 90.7]; [93.3 97.3 98.7 92 94.7 97.3]; [98.7 98.7 100 100 97.3 98.7];
                [94.7 93.3 88 92 84 94.7]; [100 100 100 98.7 97.3 100]; [100 98.7 98.7 97.3 96 97.3];
                [98.7 98.7 94.7 97.3 94.7 97.3]; [100 100 100 100 100 100]; [100 100 100 100 97.3 98.7]];
            
            tab.y_axis_range = [80 100];
            tab.colors = {[0.6 0.6 0.6], [0.6 0.6 0.6], [0.6 0.6 0.6], [0.6 0.6 0.6], ...
                [0 0.69 0.31], [0 0.69 0.31], [0 0.69 0.31], ...
                [1 0 0], [1 0 0], [1 0 0]};
            tab.line_widths = repmat(0.3, 1, numel(tab.approaches)); tab.line_widths(end-1) = 1.5;
            tab.line_specs = {':*', '--*', '-*', '-.d', '--+', '-+', '-.d', '--o', '-o', '-.d'};
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function tab = get_example3()
            % GET_EXAMPLE3: Get an example table object.
            %
            % Notes
            % -----
            % i.e. Table Object Structure
            % NLCTV + AutoNet:  [94.6 95.7 98.9]
            % NLCTV:            [90.2 91.9 97.8]
            % Glasses:          [89.2 90.9 97.1]
            %
            
            % Imports
            import nnf.utl.plot.PlotMan
            import nnf.utl.plot.Table;
            
            tab = Table('caption', 'tab:chap4:XDB', '');
            tab.approaches = {'Glasses', 'NLCTV', 'NLCTV + AutoDNet'};
            tab.values = [...
                [89.2 90.9 97.1]; [90.2 91.9 97.8]; [94.6 95.7 98.9]];
            
            tab.y_axis_range = [80 100];
            tab.colors = {[0.6 0.6 0.6], [0 0.69 0.31], [1 0 0]};
            tab.line_widths = repmat(0.3, 1, numel(tab.approaches)); tab.line_widths(end) = 1.5;
            tab.line_specs = {':*', '--*', '-*', '-.d', '--+', '-+', '-.d', '--o', '-o', '-.d'};
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = Table(caption, label, title)
            
            if (nargin < 1); caption = 'no-caption'; end
            if (nargin < 2); label = 'no-label'; end
            if (nargin < 3); title = 'no-title'; end
            
            self.caption = caption;
            self.label = label;
            self.title = title;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = validate(self)
            
            % Approach validation
            if isempty(self.approaches); success = false; return; end
            
            % Values validation
            n_approaches = numel(self.approaches, 1);
            if isempty(self.values)
                if (n_approaches ~= size(self.values, 1)); success = false; return; end
            end
            
            success = true;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end