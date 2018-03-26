classdef MovingWindowDataGenerator < handle
    % MovingWindowDataGenerator generates data via a moving window mechanism
    %   Use for in memory databases.
    %   Curent Support: Generates data for regression problems (Input, Target)
    
    properties (Dependent)
        IsEnd;
    end
        
    properties
        nndb_in;
        nndb_out;
    end
    
    properties
        st__;           % Image start index
        stop__;         % Whether iterator has reached its end
        step__;         % Step size for window
        wsize__;        % Window size
        cur_cls_i__;    % Current on going class index
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = MovingWindowDataGenerator(input, output)
            % input, output are row major matices. (Format.N_H)
                  
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;            
            
            [C, IA, IC] = unique(output, 'rows', 'stable');
            self.nndb_out = NNdb('Ouput', output, [], false, IC, Format.N_H);
            self.nndb_in = NNdb('Ouput', input, [], false, IC, Format.N_H);
                        
            % Initialize with the defaults
            self.st__ = 1;
            self.stop__ = true;
            self.step__ = 1;
            self.wsize__ = 1;
            self.cur_cls_i__ = 1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init(self, step_size, window_size)
            self.st__ = 1;
            self.stop__ = false;
            self.step__ = step_size;
            self.wsize__ = window_size;
            
            % Set current class index `self.cur_cls_i__`
            self.cur_cls_i__ = 1;           
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function reset(self)
            self.st__ = 1;
            self.stop__ = false;
            self.cur_cls_i__ = 1;
        end        
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [input, target] = next(self)
            
            if (self.IsEnd)
                error(['Iterator is already reached its end.']);
            end
            
            nndb = self.nndb_in;
            
            % Image end index
            en = self.st__ + self.wsize__ - 1;   
            
            % Calculated current class end 
            cur_cls_en = nndb.cls_st(self.cur_cls_i__) + uint32(nndb.n_per_class(self.cur_cls_i__))- 1;
            
            % Fix the end 
            if (en > cur_cls_en)
                % warning(sprintf(['windows size (=%d) with the current offset (=%d) '...'
                %    'will exceed the current class limit. Truncated data will be returned.'], ...
                %    self.wsize__, self.st__));
                en = cur_cls_en;                
            end           
            
            % Fetch data and target
            input = nndb.get_data_at(self.st__:en);              % Assumption, data_format = Format.N_H
            input = squareform(pdist(input, 'euclidean'));        % euclidean distance
            target = self.nndb_out.get_data_at(nndb.cls_st(self.cur_cls_i__));  % Assumption, data_format = Format.N_H
            
            % Fix image start index `self.st__` and `self.cur_cls_i__`
            if (en == cur_cls_en)
                self.st__ = en + 1;
                self.cur_cls_i__ = self.cur_cls_i__ + 1;
            else            
                self.st__ = self.st__ + self.step__;
            end
            
            % Fix `self.cur_cls_i__` for a case of a big step size
            for i=self.cur_cls_i__:nndb.cls_n
                if (nndb.cls_st(i) <= self.st__)
                    self.cur_cls_i__ = i;
                else
                    break;
                end
            end
                        
            if (en >= nndb.n)
                self.stop__ = true;
                return;
            end
            
            if (self.st__ > nndb.n)
                self.stop__ = true;
                return;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Dependant property Implementations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function yes = get.IsEnd(self)
            yes = self.stop__;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

