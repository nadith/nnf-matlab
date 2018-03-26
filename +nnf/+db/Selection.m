classdef Selection < handle
    % Selection denotes the selection paramters for a database.
	%
	
    properties (SetAccess = public)
        tr_col_indices      % Training column indices
        tr_noise_rate       % Noise rate or Noise types for `tr_col_indices`
        tr_occlusion_rate   % Occlusion rate for `tr_col_indices`
        tr_occlusion_type   % Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right)
        tr_occlusion_offset % Occlusion start offset from top/bottom/left/right corner depending on 'tr_occlusion_type'
        tr_out_col_indices  % Training target column indices
        val_col_indices     % Validation column indices
        val_out_col_indices % Validation target column indices
        te_col_indices      % Testing column indices
        te_out_col_indices  % Testing target column indices
        nnpatches           % NNPatch object array
        use_rgb             % Use rgb or convert to grayscale
        color_indices       % Specific color indices (set .use_rgb = false)
        use_real            % Use real valued database TODO: (if .normalize = true, Operations ends in real values)  % noqa E501
        scale               % Scaling factor (resize factor)
        normalize           % Normalize (0 mean, std = 1)
        histeq              % Histogram equalization
        histmatch_col_index % Histogram match reference column index
        class_range         % Class range for training database or all (tr, val, te)
        val_class_range     % Class range for validation database
        te_class_range      % Class range for testing database
        pre_process_script  % Custom preprocessing script
    end

    methods (Access = public) 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function self = Selection()
            % Constructs :obj:`Selection` instance.

            self.tr_col_indices      = [];      % Training column indices
            self.tr_noise_rate       = [];      % Rate or noise types for column index
            self.tr_occlusion_rate   = [];      % Occlusion rate for column index
            self.tr_occlusion_type   = [];      % Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right)
            self.tr_occlusion_offset = [];      % Occlusion start offset from top or left corner depending on 'tr_occlusion_type'
            self.tr_out_col_indices  = [];      % Training target column indices
            self.val_col_indices     = [];      % Validation column indices
            self.val_out_col_indices = [];      % Validation target column indices
            self.te_col_indices      = [];      % Testing column indices
            self.te_out_col_indices  = [];      % Testing target column indices
            self.nnpatches           = [];      % NNPatch object array
            self.use_rgb             = [];      % Use rgb or convert to grayscale
            self.color_indices       = [];      % Specific color indices (set .use_rgb = false)
            self.use_real            = [];      % Use real valued database TODO: (if .normalize = true, Operations ends in real values)  % noqa E501
            self.scale               = [];      % Scaling factor (resize factor)
            self.normalize           = false;   % Normalize (0 mean, std = 1)
            self.histeq              = false;   % Histogram equalization
            self.histmatch_col_index = [];      % Histogram match reference column index
            self.class_range         = [];      % Class range for training database or all (tr, val, te)
            self.val_class_range     = [];      % Class range for validation database
            self.te_class_range      = [];      % Class range for testing database
            self.pre_process_script  = [];      % Custom preprocessing script
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function sel = clone(self)
            % Imports
            import nnf.db.Selection;

            sel = Selection();
            sel.tr_col_indices      = self.tr_col_indices;
            sel.tr_noise_rate       = self.tr_noise_rate;
            sel.tr_occlusion_rate   = self.tr_occlusion_rate;
            sel.tr_occlusion_type   = self.tr_occlusion_type;
            sel.tr_occlusion_offset = self.tr_occlusion_offset;       
            sel.tr_out_col_indices  = self.tr_out_col_indices;
            sel.val_col_indices     = self.val_col_indices;
            sel.val_out_col_indices = self.val_out_col_indices;
            sel.te_col_indices      = self.te_col_indices;
            sel.te_out_col_indices  = self.te_out_col_indices;
            sel.nnpatches           = self.nnpatches;
            sel.use_rgb             = self.use_rgb;
            sel.color_indices       = self.color_indices;
            sel.use_real            = self.use_real;
            sel.scale               = self.scale;
            sel.normalize           = self.normalize;
            sel.histeq              = self.histeq;
            sel.histmatch_col_index = self.histmatch_col_index;
            sel.class_range         = self.class_range;
            sel.val_class_range     = self.val_class_range;
            sel.te_class_range      = self.te_class_range;
            sel.pre_process_script  = self.pre_process_script;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Protected Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function iseq = eq(self, sel)
            %{
            Equality of two :obj:`ImagePreProcessingParam` instances.

            Parameters
            ----------
            sel : :obj:`Selection`
                The instance to be compared against this instance.

            Returns
            -------
            bool
                True if both instances are the same. False otherwise.
            %}
            iseq = false;
            if (isequal(self.tr_col_indices, sel.tr_col_indices) && ...
                isequal(self.tr_noise_rate, sel.tr_noise_rate) && ...
                isequal(self.tr_occlusion_rate, sel.tr_occlusion_rate) && ...
                isequal(self.tr_occlusion_type, sel.tr_occlusion_type) && ...
                isequal(self.tr_occlusion_offset, sel.tr_occlusion_offset) && ...
                isequal(self.tr_out_col_indices, sel.tr_out_col_indices) && ...
                isequal(self.val_col_indices, sel.val_col_indices) && ...
                isequal(self.val_out_col_indices, sel.val_out_col_indices) && ...
                isequal(self.te_col_indices, sel.te_col_indices) && ...
                isequal(self.te_out_col_indices, sel.te_out_col_indices) && ...
                (self.use_rgb == sel.use_rgb) && ...
                isequal(self.color_indices, sel.color_indices) && ...
                (self.use_real == sel.use_real) && ...
                (self.normalize == sel.normalize) && ...
                (self.histeq == sel.histeq) && ...
                (self.histmatch_col_index == sel.histmatch_col_index) && ...
                isequal(self.class_range, sel.class_range) && ...
                isequal(self.val_class_range, sel.val_class_range) && ...
                isequal(self.te_class_range, sel.te_class_range) && ...
                numel(self.nnpatches) == numel(sel.nnpatches))
                % self.pre_process_script % LIMITATION: Cannot compare for eqaulity (in the context of serialization)
                iseq = true;
            end

            if (~iseq)
                return;
            end

            for i=1:numel(self.nnpatches)
                self_patch = self.nnpatches(i);
                sel_patch = sel.nnpatches(i);
                iseq = iseq && (self_patch == sel_patch);
                if (~iseq); break; end
            end       
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
