classdef LatexMan < handle
    % LATEXMAN represents latex manager for NNFramwork.
    %
    
    properties (SetAccess = private)
        file_id__;
        table__;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = LatexMan(filename, table)
            % Constructs a latexman object.
            %
            % Parameters
            % ----------
            % filename : string
            %     Filename to write the table.
            %
            % table : nnf.utl.plot.Table
            %     Table object to write the latex file.
            %
            
            self.table__ = table;
            self.file_id__ = fopen(filename,'w');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function write_table(self, column_major)
            % WRITE_TABLE: Write the latex table.
            % 
            % Parameters
            % ----------
            % column_major : bool
            %     Whether the latex table should be written in column major or row major format.
            %
            
            % Set defaults for arguments
            if nargin < 2; column_major = false; end
            
            header = strcat(...
                '\\begin{table}[htp]\n', ...
                '\\begin{center}\n', ...
                '\\begin{footnotesize}\n', ...
                ['\\caption{' self.table__.caption '} \\label{' self.table__.label '}\n'], ...
                '\\renewcommand{\\arraystretch}{1.5}\n', ...
                '\\begin{tabular}\n' ...
                );
                        
            column_headers = ['\\multirow{2}*{name} '];
            
            % Each row display the approach
            if ~column_major
                tab_values = self.table__.values;
                [n_rows, n_cols] = size(tab_values);
                column_headers = [column_headers repmat('&\\multirow{2}*{name} ', 1, n_cols)];
                
            else
                % Each column display the approach
                tab_values = self.table__.values';
                [n_rows, n_cols] = size(tab_values);
                
                for ci=1:n_cols
                    column_headers = [column_headers ...
                        ['&\\multirow{2}*{' self.table__.approaches{ci} '} ']];
                end
            end
            
            % Table values in the body
            body = '';
            
            % Itearate rows to write the table
            for ri=1:n_rows
                
                values = '';
                for ci=1:n_cols
                    y = tab_values(ri, ci);
                    assert(~isempty(y));
                    values = [values sprintf('&%.1f ', y)];
                end
                
                % Name of the approach
                row_name = 'name';
                
                % Each row display the approach
                if ~column_major
                    row_name = self.table__.approaches{ri};
                end
                
                body = strcat(body, ...
                    [row_name ' ' values '\\\\ \n'], ...
                    '\\hline\n');
            end
            
            header2 = strcat(...
                ['{' repmat('|c', 1, 1 + n_cols) '|}\n'], ...
                '\\hline\n', ...
                [column_headers '\\\\ \n'], ...
                [repmat('&', 1, n_cols) '\\\\ \n'], ...
                '\\hline\n'...
                );
            
            
            footer = strcat(...
                '\\end{tabular}\n', ...
                '\\end{footnotesize}\n', ...
                '\\end{center}\n', ...
                '\\vspace{3mm}\n', ...
                '\\end{table}' ...
                );
            
            fprintf(self.file_id__, [header header2 body footer]);
            fclose(self.file_id__);
        end       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end