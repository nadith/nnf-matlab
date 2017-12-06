classdef NNPatch < handle
    % NNPatch describes the patch information.
    %
    % i.e
    % Patch of 33x33 at position (1, 1)
    % nnpatch = NNPatch(33, 33, [1 1])
    % 
    % Attributes
    % ----------
    % h : int
    %     Height (Y dimension).
    % 
    % w : int
    %     Width (X dimension).
    % 
    % offset : (int, int)
    %     Position of the patch. (Y, X).
    % 
    % _user_data : :obj:`dict`
    %     Dictionary to store in memory patch databases for (Dataset.TR|VAL|TE...).
    % 
    % nnmodels : list of :obj:`NNModel`
    %     Associated `nnmodels` with this patch.
    % 
    % is_holistic : bool
    %     Whether the patch covers the whole image or not.
    % 
    % Notes
    % -----
    % When there is a scale operation for patches, the offset parameter will be invalidated
    % since it will not be updated after the scale operations.
    % refer init_nnpatch_fields()
    %   
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
        
    properties (SetAccess = public)
        w;      % Width        
        h;      % Height                 
        offset; % Position of the patch
        nnmodels;
        is_holistic;        
    end
    
    properties (SetAccess = private)
        user_data; % (internal use) Hold nndbtr, nndbval, nndbte patch databases in DL framework
    end
    
    properties (SetAccess = public, Dependent)
        id;
    end
    
    methods (Access = public)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = NNPatch(height, width, offset, is_holistic)
            % Constructs :obj:`NNPatch` instance.
            % 
            % Parameters
            % ----------
            % height : int
            %     Patch height.
            % 
            % width : int
            %     Image width.
            % 
            % offset : (int, int)
            %     Position of the patch. (Y, X).
            % 
            % is_holistic : bool
            %     Whether the patch covers the whole image or not.
            %
            % Matlab support preinitializing the object arrays
            % Ref: https://au.mathworks.com/help/matlab/matlab_oop/initialize-object-arrays.html                  
            if (nargin <= 0); return; end;
            
            % Imports
            import nnf.utl.Map;     
            
            % Initialize the variables
            self.h = height;
            self.w = width;            
            self.offset = offset;
            self.user_data = Map('uint32');
            self.nnmodels = [];
            self.is_holistic = is_holistic;  % Covers the whole image
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function add_model(self, nnmodels)
            % Add `nnmodels` for this nnmodel.
            % 
            % Parameters
            % ----------
            % nnmodels : :obj:`NNModel` or list of :obj:`NNModel`
            %     list of :obj:`NNModel` instances.
            %
            if (isvector(nnmodels))
                self.nnmodels = [self.nnmodels nnmodels];
            else
                self.nnmodels(end + 1) = nnmodels;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init_nnpatch_fields(self, pimg, db_format)
            % Initialize the fields of `nnpatch` with the information provided.
            % 
            % .. warning:: Offset field will not be set via this method.
            % 
            % Parameters
            % ----------
            % pimg : `array_like`
            %     Color image patch or raw data item.
            % 
            % db_format : nnf.db.Format
            %     Format of the database.
            %
            % TODO: offset needs to be adjusted accordingly
            if (db_format == Format.H_W_CH_N)
                [self.h, self.w, ~] = size(pimg.shape);

            elseif (db_format == Format.H_N)
                self.w = 1;
                self.h = size(pimg, 1);

            elseif (db_format == Format.N_H_W_CH)
                [self.h, self.w, ~] = pimg.shape;

            elseif (db_format == Format.N_H)
                self.w = 1;
                self.h = size(pimg, 1);
            end            
        end
                        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Special Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function isequal = eq(self, nnpatch)
            % Equality of two :obj:`NNPatch` instances.
            % 
            % Parameters
            % ----------
            % nnpatch : :obj:`NNPatch`
            %     The instance to be compared against this instance.
            % 
            % Returns
            % -------
            % bool
            %     True if both instances are the same. False otherwise.
            %
            isequal = false;
            % if ((self.h == nnpatch.h) &&
            %     (self.w == nnpatch.w) &&
            %     (self.offset == nnpatch.offset))
            %     isequal = true;
            % end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = protected)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function generate_nnmodels_(self)
            % Generate list of :obj:`NNModel` for Neural Network Patch Based Framework.
            % 
            % Returns
            % -------
            % list of :obj:`NNModel`
            %     `nnmodels` for Neural Network Patch Based Framework.
            % 
            % Notes
            % -----
            % Invoked by :obj:`NNPatchMan`.
            % 
            % Note
            % ----
            % Used only in Patch Based Framework. Extend this method to implement
            % custom generation of `nnmodels`.
            %
            assert(false);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = {?nnf.core.NNFramework})
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init_models(self, dict_iterstore, list_iterstore, dbparam_save_dirs)
            % Generate, initialize and register `nnmodels` for this patch.
            % 
            % Parameters
            % ----------
            % list_iterstore : :obj:`list`
            %     List of iterstores for :obj:`DataIterator`.
            % 
            % dict_iterstore : :obj:`dict`
            %     Dictionary of iterstores for :obj:`DataIterator`.
            % 
            % dbparam_save_dirs : :obj:`list`
            %     Paths to temporary directories for each user db-param of this
            %     `nnpatch`.
            % 
            % Notes
            % -----
            % Invoked by :obj:`NNPatchMan`.
            % 
            % Note
            % ----
            % Used only in Patch Based Framework.        
            self.add_model(self.generate_nnmodels_())

            % Assign this patch and iterstores to each model
            for i = 1:length(self.nnmodels)
                model = self.nnmodels(i);
                model.add_nnpatches(self);
                model.add_iterstores(list_iterstore, dict_iterstore);
                model.add_save_dirs(dbparam_save_dirs);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = setdefault_udata(self, ekey, value)
            % Set the default value in `_user_data` dictionary.
            % 
            % Parameters
            % ----------
            % ekey : :obj:`Dataset`
            %     Enumeration of `Dataset`. (`Dataset.TR`|`Dataset.VAL`|`Dataset.TE`|...)
            % 
            % value : :obj:`list`
            %     Default value [] or List of database for each user dbparam.
            %
            value = self.user_data.setdefault(ekey, value);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function set_udata(self, ekey, value)
            % Set the value in `_user_data` dictionary.
            % 
            % Parameters
            % ----------
            % ekey : :obj:`Dataset`
            %     Enumeration of `Dataset`. (`Dataset.TR`|`Dataset.VAL`|`Dataset.TE`|...)
            % 
            % value : :obj:`list`
            %     List of database for each user dbparam.
            %
            self.user_data(ekey) = value;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get_udata(self, ekey)
            % Get the value in `_user_data` dictionary.
            % 
            % Parameters
            % ----------
            % ekey : :obj:`Dataset`
            %     Enumeration of `Dataset`. (`Dataset.TR`|`Dataset.VAL`|`Dataset.TE`|...)
            % 
            % Returns
            % -------
            % :obj:`list`
            %     List of database for each user dbparam.       
            %
            value = self.user_data(ekey);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

    methods 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Dependant property Implementations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function value = get.id(self) 
            % Patch identification string.
            % 
            % Returns
            % -------
            % str
            %     Patch identification string. May not be unique.
            % 
                        
            if (~isempty(height) && ~isempty(width))
                value = sprintf('{%d}_{%d}_{%d}_{%d}', self.offset(0), self.offset(1), self.h, self.w);
            
            elseif (~isempty(height) && isempty(width))
                value = sprintf('{%d}_{%d}_{%d}_W', self.offset(0), self.offset(1), self.h, self.w);
            
            elseif (isempty(height) && ~isempty(width))
                value = sprintf('{%d}_{%d}_H_{%d}', self.offset(0), self.offset(1), self.h, self.w);
                
            elseif (isempty(height) && isempty(width))
                value = sprintf('{%d}_{%d}_H_W', self.offset(0), self.offset(1), self.h, self.w);
            end
        end
    end
end
