classdef MCF
    %MCF: Multi Color Fusion algorithm and varients.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - MCF.l2
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [minfo] = train(nndb, info) 
            % TODO: TRAIN: trains a MCF model.
            %
            % Parameters
            % ----------
            % nndb : nnf.db.NNdb
            %     Data object that contains the database.
            % 
            % info : struct, optional
            %     Provide additional information to perform LDA. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     info.sel = 0.1;                   % Regularizaion paramaeter
            %     info.scales = 0.1;                % Regularizaion paramaeter
            %     info.tolerance = false;           % Visualize fisher faces
            %     info.levels = false;              % Visualize fisher faces
            %     info.classifier.name = false;     % Visualize fisher faces  
            %     info.classifier.info = [];
            %
            % Returns
            % -------
            % minfo : cell array
            %     Trained model information.
         	%
            %
            % Examples
            % --------
            % import nnf.alg.MCF;
            % import nnf.utl.rgb2colors;
            % nndb_tr = rgb2colors(nndb_tr, false, false, true);
            %
            % sel = Selection();
            % sel.tr_col_indices  = [1:4];
            % sel.val_col_indices = [5];
            % sel.class_range     = [1:100];
            %
            % info = [];
            % info.sel = sel;
            % minfo = MCF.train(nndb_tr, info)
            % 
            % Test
            % sel = Selection();
            % sel.tr_col_indices = [1:5];
            % sel.te_col_indices = [6];
            % sel.class_range    = [1:100];
            % minfo.sel = sel;
            % MCF.test(nndb, minfo)
            

            % Imports
            import nnf.alg.*;
            import nnf.db.Format;
            import nnf.db.DbSlice; 
            import nnf.db.Selection;
   
            % Error handling for arguments
            if (nargin < 2), error(['ARG_ERR: `info`: mandatary field']); end
            if (~isfield(info, 'sel')), error(['ARG_ERR: selection structure `info[.sel]`: mandatary field']); end
            if (isempty(info.sel.tr_col_indices)), error(['ARG_ERR: selection `info.sel[.tr_col_indices]`:  mandatary field']); end
            if (isempty(info.sel.val_col_indices)), error(['ARG_ERR: selection `info.sel[.val_col_indices]`:  mandatary field']); end            
            sel = info.sel;
            
            % Color channels available
            colors = [1:nndb.ch];
            
            % Set defaults for arguments
            if (~isfield(info, 'tolerance')), info.tolerance = 1; end
            if (~isfield(info, 'scales'))
                scales_arr = {{[32 32], [24 24], [16 16], [12 12]}}; 
            else
                scales_arr = {info.scales};
            end
            
            % MCF level counts
            if (~isfield(info, 'levels')), info.levels = 3; end
            
            % Classifier choice
            if (~isfield(info, 'classifier')), info.classifier = []; end
            if (isempty(info.classifier)), info.classifier.name = 'LDA.l2'; end
            if (~isfield(info.classifier, 'info')), info.classifier.info = []; end
                        
            % Scales per each color channel
            scales_arr = repmat(scales_arr, 1, numel(colors));  
                        
            % Keep track of best performing resolutions and color channels in each level
            mcf_infos = cell(0, 0);
            
            % Initialize the current tolerance value
            cur_tolerance = info.tolerance;
            
            nndb_aug_tr = [];
            nndb_aug_val = [];
            best_accuracy = [];
            best_accuracy_index = 0;
            prev_best_accuracy = 0;
                
            for l=1:info.levels
                disp(['%%%%%%%%%%%%%% LEVEL:' num2str(l) ' %%%%%%%%%%%%%%%%']);
                         
                mcf_info = [];
                lvl_best_accuracy = [];
                
                for ci=1:numel(colors)
                    scales = scales_arr{ci};
                    
                    for si=1:numel(scales)
                        sel.use_rgb               = false;              
                        sel.scale                 = scales{si};
                        sel.color_indices         = colors(ci);
                        [nndb_tr, nndb_val, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel); 

                        nndb_tr.convert_format(Format.H_N);
                        nndb_val.convert_format(Format.H_N);
                        
                        if (~isempty(nndb_aug_tr))
                            assert(~isempty(nndb_aug_val));
                            nndb_tr = nndb_aug_tr.concat(nndb_tr);
                            nndb_val = nndb_aug_val.concat(nndb_val);
                        end                        
                        
                        % Calculate accuracy                        
                        accuracy = MCF.calculate__(nndb_tr, nndb_val, info);
                        disp(['CH:' num2str(colors(ci)) ' SCALE:' num2str(scales{si}) ' ACC:' num2str(accuracy)]);

                        % 'accuracy' = after combining complementary channels & scales across all levels
                        if (isempty(lvl_best_accuracy) || (accuracy > lvl_best_accuracy))
                            lvl_best_accuracy = accuracy;
                            mcf_info.accuracy = accuracy;
                            mcf_info.ch = colors(ci);
                            mcf_info.ci = ci;
                            mcf_info.scale = scales{si};
                            mcf_info.nndb_tr = nndb_tr;
                            mcf_info.nndb_val = nndb_val;
                        end
                                                
                        if (isempty(best_accuracy) || (accuracy > best_accuracy))
                            best_accuracy = accuracy;                            
                        end
                    end
                end
                             
                % If accuracy is not improving
                if (prev_best_accuracy == best_accuracy)
                    cur_tolerance = cur_tolerance - 1;
                else
                    best_accuracy_index = numel(mcf_infos) + 1;
                end
                prev_best_accuracy = best_accuracy;  
                
                if (cur_tolerance == 0)
                    break;
                end                
                                 
                % Training and validation database for next round
                nndb_aug_tr = mcf_info.nndb_tr;
                nndb_aug_val = mcf_info.nndb_val;                        
                
                disp(['=> SELECTED: CH:' num2str(mcf_info.ch) ' SCALE:' num2str(mcf_info.scale) ' ACC:' num2str(mcf_info.accuracy)]);
               
                % Assign it to tracking array
                mcf_infos{end + 1} = mcf_info;
                
                % Remove scale denoted by `mcf_info.scale` in `mcf_info.ch` cell since it has been
                % already selected
                scales = scales_arr{mcf_info.ci};
                scales = scales(cellfun(@(x) ~isequal(x, mcf_info.scale), scales, 'UniformOutput', 1));
                if (isempty(scales)) % no remaining scales are available                    
                    colors(mcf_info.ci) = []; % delete the element at index
                    scales_arr(mcf_info.ci) = []; % delete the cell at index
                else
                    % scales_arr{mcf_info.ci} = scales;
                end
                
                % If no colors to process
                if (isempty(colors))
                    break;
                end                
            end
            
            % Plot the accuracies
            if (isfield(info, 'plot') && info.plot)
                acc = zeros(1, numel(mcf_infos));
                for i=1:numel(mcf_infos)
                    acc(i) = mcf_infos{i}.accuracy;
                end  
                plot(acc);
            end
            
            % Remove the elements corresponds to non-best accuracies
            if (best_accuracy_index < numel(mcf_infos))
                mcf_infos(best_accuracy_index+1:end) = [];
            end
            
            % Set internal fields
            info.int.mcf_infos = mcf_infos;
            minfo = info;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = ?nnf.alg.Util, Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [accuracy] = calculate__(nndb_tr, nndb_te, info)
            % Imports
            import nnf.alg.*;
            
            clf_name = info.classifier.name;
            clf_info = info.classifier.info;
            
            if (strcmp(clf_name, 'PCA.l2'))
                W = PCA.l2(nndb_tr, clf_info);
                accuracy = Util.test(W, nndb_tr, nndb_te);
                        
            elseif (strcmp(info.classifier.name, 'LDA.l2'))
                W = LDA.l2(nndb_tr, clf_info);
                accuracy = Util.test(W, nndb_tr, nndb_te);
                
            % elseif (strcmp(info.classifier.name, 'SRC.l2'))
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end    
end

