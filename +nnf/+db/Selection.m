classdef Selection
    %{
    Selection denotes the selection paramters for a database.

    Attributes
    ----------
    TODO: Put this comment to a matlab compatible way

    Selection Structure (with defaults)
    -----------------------------------
    sel.tr_col_indices      = []    % Training column indices
    sel.tr_noise_rate       = []    % Rate or noise types for the above field
    sel.tr_out_col_indices  = []    % Training target column indices
    sel.val_col_indices     = []    % Validation column indices
    sel.val_out_col_indices = []    % Validation target column indices
    sel.te_col_indices      = []    % Testing column indices
    sel.te_out_col_indices  = []    % Testing target column indices
    sel.nnpatches           = []    % NNPatch object array
    sel.use_rgb             = []    % Use rgb or convert to grayscale
    sel.color_indices       = []    % Specific color indices (set .use_rgb = false)
    sel.use_real            = False % Use real valued database TODO: (if .normalize = true, Operations ends in real values)  % noqa E501
    
    sel.scale               = []    % Scaling factor (resize factor)
                                        int - Percentage of current size.
                                        float - Fraction of current size.
                                        tuple - Size of the output image.

    sel.normalize           = False % Normalize (0 mean, std = 1)
    sel.histeq              = False % Histogram equalization
    sel.histmatch_col_index = []    % Histogram match reference column index
    sel.class_range         = []    % Class range for training database or all (tr, val, te)
    sel.val_class_range     = []    % Class range for validation database
    sel.te_class_range      = []    % Class range for testing database
    sel.pre_process_script  = []    % Custom preprocessing script
    %}
    properties (SetAccess = public)
        tr_col_indices      % Training column indices
        tr_noise_rate       % Rate or noise types for the above field
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public Interface
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function self = Selection()
        % Constructs :obj:`Selection` instance.

        self.tr_col_indices      = [];      % Training column indices
        self.tr_noise_rate       = [];      % Rate or noise types for the above field
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Protected Interface
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            % self.pre_process_script % LIMITATION: Cannot compare for eqaulity (in the context of serilaization)
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
    end
end
