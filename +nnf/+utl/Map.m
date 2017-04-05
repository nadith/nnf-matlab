classdef Map < handle
    %ContainerMapEx is the wrapper for containers.Map
    %   Provide python like functionality for matlab containers.Map
    
    properties (SetAccess = private)
        underlyingMap;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = Map(key_type, value_type)            
            if (nargin < 1); key_type = 'char'; end
            if (nargin < 2); value_type = 'any'; end            
            self.underlyingMap = containers.Map('KeyType', key_type,'ValueType', value_type);            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = setdefault(self, key, value)            
            if (~isKey(self.underlyingMap, key))                
                self.underlyingMap(key) = value;
            end            
            value = self.underlyingMap(key);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = isKey(self, key)
            success = isKey(self.underlyingMap, key);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function self = subsasgn(self,s,b)
            self.underlyingMap(s.subs{1}) = b;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function debug(self, no_of_tabs)
            if (nargin < 2); no_of_tabs = 0; end
            fprintf(self.to_str(1));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        function str = to_str(self, no_of_tabs)            
            if (nargin < 2)
                offset = '';
            end
            
            key_set = keys(self.underlyingMap);
            str = '';
            
            for i=1:length(key_set)
                key = key_set{i};
                value = self.underlyingMap(key);
                
                if (isa(value, 'nnf.utl.Map'))
                    str_value = ['\n' value.to_str(no_of_tabs+1)];
                else
                    str_value = matlab.unittest.diagnostics.ConstraintDiagnostic.getDisplayableString(value);
                end
                
                offset = '';
                for j=1:no_of_tabs
                    offset = [offset '\t'];
                end
                
                kv_pair = sprintf('%sKEY:%s\n%s\tVALUE:%s', offset, key, offset, str_value);                
                str = sprintf('%s%s\n', str, kv_pair);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function value = get(self,key)
            value = [];
            if (isKey(self.underlyingMap, key))
                value = self.underlyingMap(key);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function map = copy(self)
            % Imports
            import nnf.utl.Map;           
            
            key_set = keys(self.underlyingMap);
            map = Map(self.underlyingMap.KeyType, self.underlyingMap.ValueType);
            for i=1:length(key_set)
                key = key_set{i};
                map.underlyingMap(key) = self.underlyingMap(key);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        % function value = subsref(self,s)        
        %     % TODO: t = Map(); isKey(t.underlyingMap, 'asd') fails
        %     value = [];
        %     if (~strcmp(s.type, '()'))
        %         return;
        %     end
        % 
        %     key = s.subs{1};
        %     if (~isKey(self.underlyingMap, key))
        %         value = self.underlyingMap(key);
        %     end 
        % end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         function m = plus(m1, m2)
%             NotKeys = unique([m1.underlyingMap.keys, 
% m2.underlyingMap.keys]);
%             v = cell(size(NotKeys));
%             for i = 1:numel(NotKeys)
%                 v{i} = m1.underlyingMap(NotKeys{i}) + 
% m2.underlyingMap(NotKeys{i});
%             end;
%             m = myMap(NotKeys, v);
%         end;
    end
end