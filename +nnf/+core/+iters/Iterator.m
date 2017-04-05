classdef Iterator < handle
    %Iterator: Base core iterator
    %   Keras.iterator functionality is provided.
    %
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties (SetAccess = {?nnf.core.iters.DataIterator})
        shuffle;
    end
    
    properties (SetAccess = protected)
        N;
        batch_size;               
        batch_index;
        total_batches_seen;
        lock;
        index_generator;
    end
    
    properties (SetAccess = private)
        is_in_loop;
        seed;
        index_array;        
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = Iterator(N, batch_size, shuffle, seed)
            self.N = N;
            self.batch_size = batch_size;
            self.shuffle = shuffle;
            self.seed = seed;
            self.batch_index = 0;
            self.total_batches_seen = 0;
            % self.lock = threading.Lock(); NO NEED
            % self.index_generator = self.flow_index(N, batch_size, seed); NO NEED
            
            % [LIMITATION: PYTHON-MATLAB]
            % Initialize state maintaining variables -alternative for yield in python
            self.is_in_loop = false;
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function reset(self)
            self.batch_index = 0;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [indices, current_index, current_batch_size] = flow_index(self)
            % [LIMITATION: PYTHON-MATLAB]
            N = self.N;
            batch_size = self.batch_size;
            seed = self.seed;
            
            if (~self.is_in_loop)
                % ensure self.batch_index is 0
                self.reset();
            end            

            while (true)
                if ~isempty(seed)
                    rng(seed + self.total_batches_seen)
                end            
                if (self.batch_index == 0)
                    self.index_array = [1:N];
                    if (self.shuffle)
                        self.index_array = randperm(N);
                    end
                end

                current_index = mod((self.batch_index * batch_size), N) + 1;
                if (N >= current_index + batch_size)
                    current_batch_size = batch_size;
                    self.batch_index = self.batch_index + 1;
                else
                    current_batch_size = N - current_index + 1;
                    self.batch_index = 0;
                end
                self.total_batches_seen = self.total_batches_seen + 1;
                
                % [LIMITATION: PYTHON-MATLAB] 
                % yield (index_array[current_index: current_index + current_batch_size],
                       % current_index, current_batch_size)                
               self.is_in_loop = true;
               indices = self.index_array(current_index: current_index + current_batch_size - 1);   
               return;
            end
        end
                   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % NO NEED
        % function self = iter__(self)
        %     % needed if we want to do something like:
        %     % for x, y in data_gen.flow(...):
        %     return self;
        % end
        % function next = next__(self, *args, **kwargs)
        %     renext = self.next(*args, **kwargs);
        % end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end