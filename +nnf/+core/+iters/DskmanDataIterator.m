classdef DskmanDataIterator < nnf.core.iters.DataIterator
    % `DskmanDataIterator` base class for :obj:`NNDiskMan' related iterators.
    % 
    % .. warning:: abstract class and must not be instantiated.
    % 
    % Iterate the database against the class ranges and column ranges that 
    % are defined via the :obj:`Selection` structure.
    % 
    % Attributes
    % ----------
    % cls_ranges : list of :obj:`list`
    %     Class range for each dataset. Indexed by enumeration `Dataset`.
    % 
    % col_ranges : list of :obj:`list`
    %     Column range for each dataset. Indexed by enumeration `Dataset`.
    % 
    % cls_ranges_max : list of int
    %     List of the class range maximums. Indexed by enumeration `Dataset`.
    % 
    % col_ranges_max : list of int
    %     List of the column range maximums. Indexed by enumeration `Dataset`.
    % 
    % union_cls_range : :obj:`list`
    %     List of all class ranges. The union set operation is applied here.
    % 
    % union_col_range : :obj:`list`
    %     List of all column ranges. The union set operation is applied here.
    % 
    % read_data_ : bool
    %     Whether to read the actual data.
    % 
    % Notes
    % -----
    % Union operations may result in omitting duplicate entries in class ranges or
    % column ranges. This is addressed in the code.        
    %
    % Perf for col_ranges and cls_ranges done since the lookup is performed on sorted cls_range and
    % col_range. Ref: filter_datasets_by_cls_col_idx(...)
    %
    
    properties
        cls_ranges;
        col_ranges;   
        
        cls_ranges_max;
        col_ranges_max;
        
        union_cls_range;
        union_col_range;
    end
    
    properties (SetAccess = protected)
        read_data_;
    end
    
    properties (SetAccess = private)
        % [LIMITATION: PYTHON-MATLAB]
        % State maintaining variables -alternative for yield in python
        i__;
        j__;
        is_in_loop_2__;
        
        cls_idx__;
        clses_visited__;
        dataset_count__;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = DskmanDataIterator(pp_params)
            % Constructor of the abstract class :obj:`DataIterator`.
            % 
            % Must call init_ex() to intialize the instance.
            % 
            % Parameters
            % ----------
            % pp_params : :obj:`dict`
            %     Pre-processing parameters for :obj:`ImageDataPreProcessor`.
            %
            self = self@nnf.core.iters.DataIterator(pp_params);            
            
            % List of class ranges (list of lists)
            % i.e 
            % cls_ranges[0] = cls_range
            % cls_ranges[1] = val_cls_range
            % cls_ranges[2] = te_cls_range
            self.cls_ranges = [];
            self.col_ranges = [];

            % Unions of class ranges and col ranges
            self.union_cls_range = [];
            self.union_col_range = [];
            
            % PERF: Whether to read the data
            self.read_data_ = true;
        
            % [INHERITED]: Used in __next__() to utilize the generator with yield
            self.gen_next_ = [];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init_ex(self, cls_ranges, col_ranges, read_data)
            % Imports
            import nnf.db.Dataset 

            % gen_next is ignored since using a custom 'self._gen_next' below
            % iter_param is ignored since it's not applicable in this context
            self.init([], []);
        
            % PERF: Whether to read the data
            self.read_data_ = read_data;
        
            % List of class ranges
            self.cls_ranges = cls_ranges;
            self.col_ranges = col_ranges;
            
            % List of the ranges max
            self.cls_ranges_max = self.ranges_max(cls_ranges);
            self.col_ranges_max = self.ranges_max(col_ranges);

            % Union of all class ranges and column ranges
            self.union_cls_range = self.union_range(cls_ranges, 'Class Range');
            self.union_col_range = self.union_range(col_ranges, 'Col Range');

            % [LIMITATION: PYTHON-MATLAB]
            % INHERITED: Used in __next__() to utilize the generator with yield
            % self.gen_next_ = self.next()

            % TODO: USE PERSISTANT VARIABLES (But look for thread safety)
            % [LIMITATION: PYTHON-MATLAB]
            % Initialize state maintaining variables -alternative for yield in python
            self.i__ = 1;
            self.is_in_loop_2__ = false;            
            self.cls_idx__ = -1;  % will be set in loop_1
            self.dataset_count__ = uint8(zeros(1, numel(self.col_ranges)));  % Dataset served count 

            % Track the classes to decide newly added classes for class ranges
            % Keyed by the range index of class ranges
            % TODO: HACK due to limitation of matlab of Map container definition for int key.
            % Ref:
            self.clses_visited__ = containers.Map(uint32(Dataset.TR), uint16(zeros(2)));
            remove(self.clses_visited__, uint32(Dataset.TR));
        end       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [cimg, frecord, cls_idx, col_idx, filtered_entries, stop] = next(self)
            % Generate the next valid image
            %
            % IMPLEMENTATION NOTES:        
            %     Column Repeats
            %     --------------
            %         sel.xx_col_indices = [1 1 1 3] handled via yield repeats
            % 
            %     Class Repeats
            %     --------------
            %     LIMITATION: sel.cls_range = [0, 0, 0, 2] is not supported
            %         Reason: since the effect gets void due to the union operation
            %         TODO:   Copy and append sel.xx_col_indices to itself when 
            %                 there is a class repetition        
            %
            
            % Import 
            import nnf.db.Dataset
            import nnf.db.Select
            
            % Local variables
            stop = false;
            
            % Itearate the union of the class ranges i.e (tr, val, te, etc)                        
            while (true)

                if (~self.is_in_loop_2__)
                
                    if (isscalar(self.union_cls_range) && isa(self.union_cls_range, 'Select'))
                        self.cls_idx__ = self.i__;

                        % Update loop index variable
                        self.i__ = self.i__ + 1;

                        % When the first col_idx is out of true range, break
                        if (~self.is_valid_col_idx_(self.cls_idx__, false))
                            self.check_and_issue_warning(self.cls_idx__, self.cls_ranges_max, ... 
                                ['Class: >=' num2str(self.cls_idx__) ' are missing in the database']);
                            break;
                        end
                        
                    else
                        if (self.i__ > numel(self.union_cls_range)); break; end
                        self.cls_idx__ = self.union_cls_range(self.i__);

                        % Update loop index variable
                        self.i__ = self.i__ + 1;

                        % When a cls_idx is out of true range, skip
                        if (~self.is_valid_cls_idx_(self.cls_idx__)); continue; end
                    end

                    % Itearate the union of the column ranges i.e (tr, val, te, etc)
                    self.j__ = 1;
                end

                while (true)

                    % Save state
                    self.is_in_loop_2__ = true;
                    
                    if (isscalar(self.union_col_range) && isa(self.union_col_range, 'Select'))
                        col_idx = self.j__;

                        % Update loop index variable
                        self.j__ = self.j__ + 1;

                        % When the col_idx is out of true range, break
                        if (~self.is_valid_col_idx_(self.cls_idx__, col_idx, false))
                            self.check_and_issue_warning(col_idx, self.col_ranges_max, ...
                                ['Class:' num2str(self.cls_idx__) ' ImageIdx: >=' ...
                                num2str(col_idx) ' are missing in the database'])
                            break;
                        end

                    else                        
                        if (self.j__ > numel(self.union_col_range)); break; end
                        col_idx = self.union_col_range(self.j__);

                        % Update loop index variable
                        self.j__ = self.j__ + 1;

                        % When a col_idx is out of true range, skip
                        if (~self.is_valid_col_idx_(self.cls_idx__, col_idx))
                            continue;
                        end
                    end

                    % Filter datasets by class index (cls_idx) and coloumn index (col_index).
                    % filtered_datasets => [(TR, is_new_class), (VAL, ...), (TE, ...), ...]
                    filtered_entries = ...
                            self.filter_datasets_by_cls_col_idx( ...
                                                            self.cls_idx__, ...
                                                            col_idx, ...
                                                            (self.i__- 1), ...
                                                            self.clses_visited__, ...
                                                            self.dataset_count__);

                    % Validity of col_idx in the corresponding self.cls_ranges[rng_idx]
                    if (numel(filtered_entries) == 0); continue; end

                    % Fetch the image at cls_idx, col_idx
                    [cimg, frecord] = self.get_cimg_frecord_in_next_(self.cls_idx__, col_idx);
                    cls_idx = self.cls_idx__;
                    return;

                    % TODO: Use self._imdata_pp to pre-process data

                    % [LIMITATION: PYTHON-MATLAB]
                    % yield cimg, frecord, cls_idx, col_idx, filtered_entries  # all_entries
                end

                self.is_in_loop_2__ = false;                
            end

            cimg = [];
            frecord = [];
            cls_idx = -1;
            col_idx = -1;
            filtered_entries = [];
            stop = true;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = public) % `protected` is invalid, DbSlice, etc will not have access !
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function release(self)
            % Release internal resources used by the iterator.
            % release@nnf.core.iters.DataIterator(pp_params);
            self.cls_ranges = [];
            self.cls_ranges_max = [];
        	self.col_ranges_max = [];
            self.union_cls_range = [];
            self.union_col_range = [];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Abstract, Access = protected)
        [cimg, frecord] = get_cimg_frecord_in_next_(self, cls_idx, col_idx)  % Fetch image @ cls_idx, col_idx
        valid = is_valid_cls_idx_(self, cls_idx, show_warning)  % Check the validity cls_idx
        valid = is_valid_col_idx_(self, cls_idx, col_idx, show_warning)  % Check the validity col_idx of the class denoted by cls_idx
        valid = get_n_per_class_(self, cls_idx)  % Fetch no. of samples per class
    end
    
    methods (Access = private)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function ranges_max = ranges_max(self, ranges)
            % Imports 
            import nnf.db.Select
            
            % Build ranges max array
            ranges_max = uint8(zeros(1, numel(ranges)));

            for ri=1:numel(ranges)
                % range can be enum-Select.ALL|... or numpy.array([])
                range = ranges{ri};
                
                if (~iscell(range))
                    if (~isempty(range) && ... 
                        (~(isscalar(range) && isa(range, 'Select'))))
                        ranges_max(ri) = max(range);
                    else
                        ranges_max(ri) = 0;
                    end
                    
                else
                    % range = {[1 2 3], [4 6]};
                    for i=1:numel(range)
                        range_vec = range{i};
                        if (~isempty(range_vec) && ... 
                            (~(isscalar(range_vec) && isa(range_vec, 'Select'))))
                            ranges_max(ri) = max([ranges_max(ri) range_vec]);
                        else
                            ranges_max(ri) = 0;
                        end
                    end
                end
                    
            end
        end
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [res] = union_range(self, ranges, name)
            % Build union of ranges
            
            % Imports
            import nnf.db.Select;
            import nnf.db.Dataset;
            
            % Union of ranges
            res = [];       
            for ri=1:numel(ranges)
                range = ranges{ri};
                
                % range can be None or enum-Select.ALL or numpy.array([])
                if (isempty(range))
                    continue;
                end

                if (~iscell(range))
                    if (isscalar(range) && isa(range, 'Select'))
                        res = range;
                        return
                    end
                    
                    % Issue a warning if user-required order is going to be altered
                    tmp = sort(range);
                    if (~isequal(tmp, range))
                        warning([Dataset.str(Dataset.enum(ri)) ' ' name ' is not in sorted order.']);
                    end
                    
                    res = union(res, range);
                    
                else
                    % range = {[1 2 3], [4 6]};
                    for i=1:numel(range)
                        range_vec = range{i};
                        if (isscalar(range_vec) && isa(range_vec, 'Select'))
                            res = range_vec;
                            return
                        end
                        
                        % Issue a warning if user-required order is going to be altered
                        tmp = sort(range_vec);
                        if (~isequal(tmp, range_vec))
                            warning([Dataset.str(Dataset.enum(ri)) ' ' name ' is not in sorted order.']);
                        end
                        
                        res = union(res, range_vec);
                    end
                end
            end 
            
            % union with a empty list returns a column vector
            res = res';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [success, dataset_count] = update_dataset_count(self, dataset_count, ri, cls_idx, col_range)
            % Track dataset counters to process special enum Select.ALL|... values
            c = dataset_count(ri);

            % Set defaults
            if (nargin < 5)
                col_range = [];
            end

            if (~isempty(col_range))                                
                if (isscalar(col_range) && isa(col_range, 'Select'))
                    ratio = 1;
                    if (col_range == Select.PERCENT_40)
                        ratio = 0.4;
                    elseif (col_range == Select.PERCENT_60)
                        ratio = 0.6;
                    end

                    if (c >= self.get_n_per_class_(cls_idx)*ratio)
                        success = false;
                    end
                end
            end

            dataset_count(ri) = c + 1;
            success = true;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function cls_not_visited = update_clses_visited(self, clses_visited, edataset, cls_idx)
            % Track classes that are already visited for each dataset
            % Keep track of classes already visited
            cls_not_visited = true;
            
            if (~isKey(clses_visited, uint32(edataset)))
                clses_visited(uint32(edataset)) = [];
            end
            
            % List of classes visited
            clses_visited_at_dataset = clses_visited(uint32(edataset));
            
            for cls_visited = clses_visited_at_dataset
                if (cls_visited == cls_idx)
                    cls_not_visited = false;
                    break;
                end
            end

            % Update classes visited
            if (cls_not_visited)
                clses_visited_at_dataset(end+1) = cls_idx;
            end
            
            % Update the dict
            clses_visited(uint32(edataset)) = clses_visited_at_dataset;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function filtered_entries = filter_datasets_by_cls_col_idx( ...
                                                            self, ...
                                                            cls_idx, ...
                                                            col_idx, ...
                                                            cls_counter, ...
                                                            clses_visited, ...
                                                            dataset_count)
            % Filter range indices by class index (cls_idx) and column index (col_index).
            % 
            % Handle repeated columns as well as processing special enum values. i.e Select.ALL|...
            % 
            % Parameters
            % ----------
            % cls_idx : int
            %     Class index. Belongs to `union_cls_range`.
            % 
            % col_idx : int
            %     Column index. Belongs to `union_col_range`.
            % 
            % clses_visited : :obj:`dict`
            %     Keep track of classes already visited for 
            %     each dataset (Dataset.TR|VAL|TE...)
            % 
            % Returns
            % -------
            % list of :obj:`tuple`
            %     Each `tuple` consist of enumeration `Dataset` and bool indicating
            %     a new class addition.
            %     i.e [(Dataset.TR, is_new_class), (Dataset.VAL, True), (Dataset.VAL, False), ...]
            % 
            % Notes
            % -----
            % if.
            % col_ranges[Dataset.TR.int()] = [0 1 1]
            % col_ranges[Dataset.VAL.int()] = [1 1]
            % cls_ranges[Dataset.TR.int()] = [0 1]
            % cls_ranges[Dataset.VAL.int()] = [1]
            % 
            % then.
            % cls_idx=0 iteration, 
            % (col_idx=0, [(Dataset.TR, True)]): => [(Dataset.TR, True)]
            % 
            % cls_idx=0 iteration, 
            % (col_idx=1, [(Dataset.TR, False)]): => [(Dataset.TR, False), (Dataset.TR, False)]
            % 
            % cls_idx=1 iteration,
            % (col_idx=0, [(Dataset.TR, True)]): => [(Dataset.TR, True)]
            % 
            % cls_idx=1 iteration,
            % (col_idx=1, [(Dataset.TR, False), (Dataset.VAL, True)]): 
            %             => [(Dataset.TR, False), (Dataset.TR, False), (Dataset.VAL, True), (Dataset.VAL, False)]
            %
                    
            % Imports
            import nnf.db.Dataset
            import nnf.db.Select
            
            filtered_entries = cell(0, 0);

            % Iterate through class ranges (i.e cls_range, val_cls_range, te_cls_range, etc)
            for ri=1:numel(self.cls_ranges)                
                
                % cls_range can be None or enum-Select.ALL or numpy.array([])
                cls_range = self.cls_ranges{ri};
                if (isempty(cls_range)); continue; end

                if ((isscalar(cls_range) && isa(cls_range, 'Select')) || ...
                    (numel(intersect(cls_idx, cls_range)) ~= 0))                

                    % col_range can be enum-Select.ALL or []
                    col_range = self.col_ranges{ri};                   
                    if (isempty(col_range)); continue; end

                    if (iscell(col_range))
                        % col_range = {[1 2 3], [4 6]};
                        col_range = col_range{cls_counter};
                    end 
                
                    % Dataset.TR|VAL|TE|...
                    edataset = Dataset.enum(ri);

                    % First, Update counters at dataset
                    [success, dataset_count] = self.update_dataset_count(dataset_count, ...
                                                        ri, ...
                                                        cls_idx, ...
                                                        col_range);
                    if (~success); continue; end                        
                    
                    % Then, Add the entry
                    if (isscalar(col_range) && isa(col_range, 'Select'))
                        cls_not_visited = self.update_clses_visited(clses_visited, edataset, cls_idx);
                        dataset_tup = {edataset, cls_not_visited};  % Entry
                        filtered_entries{end+1} = dataset_tup;

                    % Or, Add the entry while adding duplicates for repeated columns
                    else
                        is_first = true;
                        for ci = sort(col_range)  % PERF
                            if (ci == col_idx)
                                cls_not_visited = false;
                                if (is_first)
                                    cls_not_visited = ...
                                        self.update_clses_visited(clses_visited, edataset, cls_idx);
                                    is_first = false;
                                end

                                dataset_tup = {edataset, cls_not_visited};  % Entry
                                filtered_entries{end+1} = dataset_tup;
                            end

                            if (ci > col_idx); break; end  % PERF
                        end                        
                    end
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function check_and_issue_warning(self, idx, ranges_max, msg)
            % Check and issue a warning if `ranges_max` are invalid.
            for ri=1:numel(ranges_max)
                rmax = ranges_max(ri);
                if (idx <= rmax)
                    warning(msg)
                    break;
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end   
  
end
