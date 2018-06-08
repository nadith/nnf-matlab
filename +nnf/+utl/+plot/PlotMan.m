classdef PlotMan < handle
    % PLOTMAN represents plot manager for NNFramwork.
    %   Draws plots from a table object.
    
    properties (SetAccess = private)
        table;
        params;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = PlotMan(table, params)
            % Constructs a plotman object.
            %
            % Parameters
            % ----------
            % info : struct, optional
            %     Provide additional information to plotman plots. (Default value = []).
            %
            %     Params Structure (with defaults)
            %     -----------------------------------
            %     inf.legend_pos = 'south';             % Position of the legend
            %     inf.x_label = 'Test Case Index';      % X-axis label
            %     inf.y_label = 'Accuracy';             % Y-axis label
            %     inf.font_size.axis_marker = 10;       % Fontsize of axis markers
            %     inf.font_size.axis_label = 10;        % Fontsize of axis labels
            %     inf.font_size.legend = 9;             % Fontsize of legend
            %     inf.font_size.title = 12;             % Fontsize of the figure title
            %     inf.font_size.marker = 8;             % Fontsize of markers
            %
            % table : nnf.utl.plot.Table
            %     Table object to write the latex file.
            %
            
            self.table = table;
            
            % Set defaults for arguments
            if nargin < 2 || isempty(params); params = struct; end
            if ~isfield(params, 'legend_pos'); params.legend_pos = 'south'; end
            if ~isfield(params, 'x_label'); params.x_label = 'Test Case Index'; end
            if ~isfield(params, 'y_label'); params.y_label = 'Accuracy'; end
            
            if ~isfield(params, 'font_size'); params.font_size = struct; end
            if ~isfield(params.font_size, 'axis_marker'); params.font_size.axis_marker = 10; end
            if ~isfield(params.font_size, 'axis_label'); params.font_size.axis_label = 10; end
            if ~isfield(params.font_size, 'legend'); params.font_size.legend = 9; end
            if ~isfield(params.font_size, 'title'); params.font_size.title = 12; end
            if ~isfield(params.font_size, 'marker'); params.font_size.marker = 8; end
            
            self.params = params;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function plot(self)
            
            % Error handling
            if ~(self.table.validate())
                error('Table validation is failed.');
            end
            
            % Change default axes fonts.
            set(0,'DefaultAxesFontName', 'Times New Roman')
            set(0,'DefaultAxesFontSize', self.params.font_size.axis_marker)
            
            % Change default text fonts.
            set(0,'DefaultTextFontname', 'Times New Roman')
            % set(0,'DefaultTextFontSize', 30) % unsed atm
            
            max_columns = size(self.table.values, 2);
            
            % Itearate rows to plot the graph
            for fi=1:numel(self.table.approaches)
                
                y = self.table.values(fi, :);
                assert(~isempty(y));
                
                color = 'red';
                if ~isempty(self.table.colors)
                    color = self.table.colors{fi};
                end
                
                line_width = 1;
                if ~isempty(self.table.line_widths)
                    line_width = self.table.line_widths(fi);
                end
                
                line_spec = '-';
                if ~isempty(self.table.line_specs)
                    line_spec = self.table.line_specs{fi};
                end
                
                self.plot__(1:max_columns, y, color, line_spec, line_width);
                h = xlabel(self.params.x_label);
                h.FontSize = self.params.font_size.axis_label;
                h = ylabel(self.params.y_label);
                h.FontSize = self.params.font_size.axis_label;
                
                hold on;
            end
            
            % Legend settings
            lgnd = legend(self.table.approaches, 'Location', self.params.legend_pos);
            lgnd.FontSize = self.params.font_size.legend;
            set(lgnd,'color','none');
            
            % Figure title settings
            h = title(self.table.title);
            h.FontSize = self.params.font_size.title;
            
            hold off;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function plot__(self, x, y, color, line_spec, line_width)
            if (isempty(y)); return; end
            
            % Matlab plot method
            pp = plot(x, y, line_spec, 'LineWidth', line_width, 'MarkerSize',5, 'MarkerFaceColor', [1 1 1], 'Color', color);
            
            % Write text markup below the y-value points
            for i=1:numel(y)
                y_val = y(i);
                h = text(i + 0.05, y_val - 0.5, num2str(y_val));
                h.FontSize = self.params.font_size.marker;
                h.Color = color;
            end
            
            % h = text(x(1), y(1)+1, sprintf(' \\leftarrow %s', title));
            % h.FontSize = 8;
            % h.Color = color;
            
            % X and Y-Axis settings
            if (numel(x) > 1)
                if (~isempty(self.table.y_axis_range))
                    axis([1 numel(x) self.table.y_axis_range]);
                else
                    axis([1 numel(x) 0 100]);
                end
                
                xData = get(pp, 'XData');
                set(gca, 'Xtick', linspace(xData(1),xData(end),numel(y)));
                
            else
                if (~isempty(self.table.limit_y))
                    axis([0 2 self.limit_y]);
                else
                    axis([0 2 0 100]);
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end