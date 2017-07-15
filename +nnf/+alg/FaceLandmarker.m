classdef FaceLandmarker
    %FaceLandmarker: Detect landmarks for face image.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - FaceLandmarker.intraface (intra-face landmark detector)
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [landmarks, pose, im_fail] = intraface(img, is_cropped, info)
            % intraface: intra-face landmarking for face.   
            %
            % Parameters
            % ----------
            % img : matrix -uint8
            %     Face image to be landmarked.
            %
            % is_cropped : bool, optional
            %     Whether the img is already cropped. (Default value = false).
            %
            % info : struct, optional
            %     Provide additional information to perform landmark detection. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     info.show_im_fail = true;     % show image at failure
            %     info.vj.MinNeighbors = 4;     % Minimum neighbours for viola jones (Matlab)
            %     info.vj.ScaleFactor = 1.1;    % Scale factor for viola jones (Matlab)
            %     info.vj.MinSize = [20 20];    % Minimum size for viola jones (Matlab)
            %
            % Returns
            % -------
            % landmarks : matrix -double
            %     49 Landmarks indicating [x, y] cordinates.
         	%
            % pose : struct
            %     pose.angle  1x3 - rotation angle, [pitch, yaw, roll]
            %     pose.rot    3x3 - rotation matrix
         	%
            
            % Imports
            import nnf.alg.FaceLandmarker;            
            
            landmarks = [];
            pose = [];
            im_fail = [];
            show_im_fail = true;
            vj = struct;
            if (nargin < 2), is_cropped = false; end
            if (nargin < 3), info = []; end
            if (isfield(info,'show_im_fail')); show_im_fail = info.show_im_fail; end
            if (isfield(info,'vj')); vj = info.vj; end
            if (~isfield(vj,'MinNeighbors')); vj.MinNeighbors = 4; end
            if (~isfield(vj,'ScaleFactor')); vj.ScaleFactor = 1.1; end
            if (~isfield(vj,'MinSize')); vj.MinSize = [20 20]; end
            
            % Disable figure visibility
            if (~show_im_fail)
                set(0,'DefaultFigureVisible','off');
            end
            
            % load model and parameters, type 'help xx_initialize' for more details
            [Models,option] = xx_initialize;

            % Viola jones
            faces = Models.DM{1}.fd_h.detect(img, 'MinNeighbors', vj.MinNeighbors,...%option.min_neighbors,...
              'ScaleFactor', vj.ScaleFactor, 'MinSize', vj.MinSize);
            % figure, imshow(img), hold on;

            % If viola-jhones fails for a already cropped face
            if (isempty(faces) && is_cropped)  
                
                % Attempt full box
                [landmarks, box, pose] = FaceLandmarker.process_cropped__(Models, img, option);                
                if isempty(landmarks)
                    figure, imshow(img); hold on
                    rectangle('Position', box, 'EdgeColor','r'); hold off;
                    [im_fail, ~] = frame2im(getframe(gca));
                    warning('Landmark detection error');
                end
                
            elseif (~isempty(faces))                
                % Assume only 1 face is there in the image
                % for i = 1:length(faces)                
                    output = xx_track_detect(Models, img, faces{1}, option);
                    
                    % Intraface may fail for a false detection of viola jones
                    if ~isempty(output.pred)
                        pose = output.pose;
                        landmarks = output.pred;
                        % plot(output.pred(:,1),output.pred(:,2),'g*','markersize',2); hold off;

                        % % for kk = 1:length(output.pred(:,1))
                        % %     text(double(output.pred(kk,1)),double(output.pred(kk,2)),num2str(kk),'Color','w','FontSize', 20);
                        % % end
                        
                    else
                        % Attempt full box
                        [landmarks, box, pose] = FaceLandmarker.process_cropped__(Models, img, option);
                        if isempty(landmarks)
                            figure, imshow(img); hold on
                            rectangle('Position', faces{1}, 'EdgeColor', 'g'); hold on;
                            rectangle('Position', box, 'EdgeColor','r'); hold off;
                            [im_fail, ~] = frame2im(getframe(gca));
                            warning('Landmark detection error');                    
                        end                        
                    end                    
                % end
                
            else
                figure, imshow(img);
                [im_fail, ~] = frame2im(getframe(gca)); % Will return color channels
                warning('Viola-Jones fails to detect the face');
                warning('Landmark detection error');                
            end
            
            % Restore
            if (~show_im_fail)
                set(0,'DefaultFigureVisible','on');
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [landmarks, box, pose] = process_cropped__(Models, img, option)
            box = [1 1 size(img, 2)-1 size(img, 1)-1];
            output = xx_track_detect(Models, img, box, option);
            if ~isempty(output.pred)
                pose = output.pose;
                landmarks = output.pred;
            else
                pose = [];
                landmarks = [];                   
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end    
end

