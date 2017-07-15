classdef FaceAligner
    %FaceAligner: Align and crop face with landmarks.
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - FaceAligner.align
    
    % Copyright 2015-2016 Nadith Pathirage, Curtin University (chathurdara@gmail.com).
    
    properties
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndb_aln, nndb_fail, fail_cls_lbl] = align(nndb_or_path, is_cropped, info) 
            % align: align and crop the faces with landmarks.   
            %
            % Parameters
            % ----------
            % nndb_or_path : nnf.db.NNdb or `string`
            %     NNdb or path to the image directory.
            %     If image directory, the images must be arranged in folders named in class names.
            %     If NNdb, the images are assumed to be cropped for landmark detection.
            %
            % is_cropped : bool, optional
            %     Whether the images are considered already cropped. (Default value = false).
            %     Violajones may fail for cropped images. Incase it fails, it feeds the whole image
            %     if (is_cropped == true).
            %
            % info : struct, optional
            %     Provide additional information to perform alignment. (Default value = []).    
            %        
            %     Info Structure (with defaults)
            %     -----------------------------------
            %     info.landmarks = [];                  % (no_of_landmarks x 2) matrix 
            %     info.save_path = [];                  % Path to save aligned images
            %
            %     info.desired_face: properties
            %     -----------------------------------
            %     info.desired_face.height = 66;       % imheight after alignment and crop
            %     info.desired_face.width = 66;        % imwidth after alignment and crop
            %     info.desired_face.lex_ratio = 0.25;   % left eye centre `x` position ratio w.r.t  
            %                                           % top, left; calculated from desired width.
            %     info.desired_face.rex_ratio = 0.75;   % right eye centre `x` position ratio w.r.t
            %                                           % top, left; calculated from desired width.
            %     info.desired_face.ey_ratio = 0.33;    % both eyes centre `y` position ratio w.r.t
            %                                           % top, left; calculated from desired height.
            %     info.desired_face.my_ratio = 0.63;    % mouth center `y` position ratio w.r.t
            %                                           % top, left; calculated from desired height.
            %     info.detector: properties
            %     ------------------------------------                                 
            %     info.detector.show_im_fail = true;    % show image at detector failure
            %     info.detector.vj.MinNeighbors = 4;    % Minimum neighbours for viola jones (Matlab)
            %     info.detector.vj.ScaleFactor = 1.1;   % Scale factor for viola jones (Matlab)
            %     info.detector.vj.MinSize = [20 20];   % Minimum size for viola jones (Matlab)
            %            
            % Returns
            % -------
            % nndb_aln : nnf.db.NNdb
            %     NNdb object with aligned images.
         	%
            % Examples
            % --------
            % import nnf.alg.FaceAligner;
            % info.save_path = 'D:\TEST_FOLDER';
            % nndb_alg = FaceAligner.align('C:\images', info);
            %                   or
            % nndb_alg = FaceAligner.align(nndb, info);
            %

            % Imports
            import nnf.db.NNdb;
            import nnf.alg.FaceAligner;
                        
            dir_db = []; nndb = [];
            save_path = [];
            
            % Initialize variables
            if (isa(nndb_or_path, 'char'))
                dir_db = nndb_or_path;
            elseif (isa(nndb_or_path, 'NNdb'))
                nndb = nndb_or_path;
            else
                error('`nndb_or_path` unknown: NNdb or path to the image directory is expected');
            end
            
            % Set defaults for arguments
            if (nargin < 2), is_cropped = false; end
            if (nargin < 3), info = []; end           
            if (isfield(info,'save_path')); save_path = info.save_path; end
            
            % Make a new directory to save the aligned images
            if (~isempty(save_path) && exist(save_path, 'dir') == 0)
                mkdir(save_path);
            end               
                            
            % If images need to be fetched from directory
            if (~isempty(dir_db))
                [nndb_aln, nndb_fail, fail_cls_lbl] = ...
                    FaceAligner.process_db_dir__(dir_db, is_cropped, save_path, info);
                
            elseif (~isempty(nndb))
                [nndb_aln, nndb_fail, fail_cls_lbl] = ...
                    FaceAligner.process_nndb__(nndb, is_cropped, save_path, info);
            end            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [detector, landmarks, desired_face] = unpack_params__(info)            
            % Set defaults for arguments
            detector = [];            
            landmarks = [];
            desired_face = [];
            if (isfield(info,'detector')); detector = info.detector; end            
            if (isfield(info,'landmarks')); landmarks = info.landmarks; end 
            if (isfield(info,'desired_face')); desired_face = info.desired_face; end 
            if (isempty(desired_face))
                desired_face.height = 66;
                desired_face.width = 66;
                desired_face.lex_ratio = 1/4;
                desired_face.rex_ratio = 3/4;
                desired_face.ey_ratio = 0.33;
                desired_face.my_ratio = 0.63; % preserve chin
            else
                if (~isfield(desired_face,'height')); desired_face.height = 66; end
                if (~isfield(desired_face,'width')); desired_face.width = 66; end
                if (~isfield(desired_face,'lex_ratio')); desired_face.lex_ratio = 1/4; end
                if (~isfield(desired_face,'rex_ratio')); desired_face.rex_ratio = 3/4; end
                if (~isfield(desired_face,'ey_ratio')); desired_face.ey_ratio = 0.33; end
                if (~isfield(desired_face,'my_ratio')); desired_face.my_ratio = 0.63; end
            end 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndb_aln, nndb_fail, fail_cls_lbl] = process_db_dir__(dir_db, is_cropped, save_path, info)

            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
            import nnf.alg.FaceLandmarker;
            import nnf.alg.FaceAligner;
            
            % Set defaults for arguments
            [detector, landmarks, desired_face] = FaceAligner.unpack_params__(info);
            
            % Init empty NNdb to collect images
            nndb_aln = NNdb('ALIGNED', [], [], false, [], Format.H_W_CH_N);
            nndb_fail = NNdb('FAILED', [], [], false, [], Format.H_W_CH_N);
            
            % Class label corresponds to original db
            fail_cls_lbl = [];
            
            cls_structs = dir(dir_db);
            cls_structs = cls_structs(~ismember({cls_structs.name},{'.','..'})); % exclude '.' and '..'

            % Sort the folder names (class names)
            [~,ndx] = natsortfiles({cls_structs.name}); % indices of correct order
            cls_structs = cls_structs(ndx);             % sort structure using indices
            
            % Iterator
            for cls_i = 1:length(cls_structs)

                cls_name = cls_structs(cls_i).name;
                cls_dir = fullfile(dir_db, cls_name);

                % Make a directory to save the images of this class                    
                if (~isempty(save_path) && exist(fullfile(save_path, cls_name), 'dir') == 0)
                    mkdir(fullfile(save_path, cls_name));
                end

                % img_structs = dir (fullfile(ims_dir, '*.jpg')); % Only jpg files
                img_structs = dir (cls_dir);
                img_structs = img_structs(~ismember({img_structs.name},{'.','..'})); % exclude '.' and '..'

                % Sort the image files (file names)
                [~,ndx] = natsortfiles({img_structs.name}); % indices of correct order
                img_structs = img_structs(ndx);             % sort structure using indices            
            
                is_new_class = true;
                is_new_class_for_fail = true; 

                for cls_img_i = 1 : length(img_structs)
                    img_name = img_structs(cls_img_i).name;
                    img = imread(fullfile(cls_dir, img_name));

                    % Fetch or detect landmarks
                    if (~isempty(landmarks))
                        lmarks_img = landmarks{cls_img_i};
                    else
                        [lmarks_img, pose, im_fail] = FaceLandmarker.intraface(img, is_cropped, detector);
                    end

                    % If landmarking fails
                    if (isempty(lmarks_img))
                        if (isempty(nndb_fail.db) || ((size(nndb_fail.db, 1) == size(im_fail, 1)) && ... % height
                                (size(nndb_fail.db, 2) == size(im_fail, 2)) && ... % width
                                (size(nndb_fail.db, 3) == size(im_fail, 3)))) % channels

                            % Update NNdb for failed images
                            nndb_fail.add_data(im_fail);
                            nndb_fail.update_attr(is_new_class_for_fail);                        
                            is_new_class_for_fail = false;
                            fail_cls_lbl = cat(2, fail_cls_lbl, uint16([str2num(cls_name)]));
                        else
                            warning('nndb_fail: Failed images are in different resolutions, ignored...')
                        end                                               
                        continue;
                    end                       

                    % Filter with pose
                    if (~FaceAligner.pose_filter__(pose, info))
                        continue;
                    end 

                    % Perform alignment with landmarks for img
                    img_aligned = FaceAligner.align_face__(img, lmarks_img, desired_face);

                    % Save the aligned image
                    if (~isempty(save_path))
                        [~,img_name,~] = fileparts(img_name); % Exclude file extension
                        imwrite(img_aligned, fullfile(save_path, cls_name, [img_name '_aligned.jpg']), 'jpg');
                    end

                    % Update NNdb
                    nndb_aln.add_data(img_aligned);
                    nndb_aln.update_attr(is_new_class);                        
                    is_new_class = false;                       
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [nndb_aln, nndb_fail, fail_cls_lbl] = process_nndb__(nndb, is_cropped, save_path, info)
            
            % Imports
            import nnf.db.NNdb;
            import nnf.db.Format;
            import nnf.alg.FaceLandmarker;
            import nnf.alg.FaceAligner;
            
            % Set defaults for arguments
            [detector, landmarks, desired_face] = FaceAligner.unpack_params__(info);
            
            % Init empty NNdb to collect images
            nndb_aln = NNdb('ALIGNED', [], [], false, [], Format.H_W_CH_N);
            nndb_fail = NNdb('FAILED', [], [], false, [], Format.H_W_CH_N);
          
            % Class label corresponds to original db
            fail_cls_lbl = []; 
            
            cls_img_i = 1;
            cls_i = 1; 
            is_new_class = true;
            is_new_class_for_fail = true;

            for img_i=1:nndb.n                    
                % Fetch class name 
                cls_name = num2str(nndb.cls_lbl(img_i));

                % Make a directory to save the images of this class                    
                if (~isempty(save_path) && exist(fullfile(save_path, cls_name), 'dir') == 0)
                    mkdir(fullfile(save_path, cls_name));
                end 

                % Determine image id
                if (cls_i < numel(nndb.cls_st) && (nndb.cls_st(cls_i + 1) <= img_i))
                    cls_i = cls_i + 1;
                    is_new_class = true;
                    is_new_class_for_fail = true;
                    cls_img_i = 1;
                else
                    cls_img_i = cls_img_i + 1;
                end

                % Fetch image @ index `img_i`
                img = nndb.get_data_at(img_i);

                % Fetch or detect landmarks
                if (~isempty(landmarks))
                    lmarks_img = landmarks{img_i};
                else
                    [lmarks_img, pose, im_fail] = FaceLandmarker.intraface(img, is_cropped, detector);
                end

                % If landmarking fails
                if (isempty(lmarks_img))
                    if (isempty(nndb_fail.db) || ((size(nndb_fail.db, 1) == size(im_fail, 1)) && ... % height
                            (size(nndb_fail.db, 2) == size(im_fail, 2)) && ... % width
                            (size(nndb_fail.db, 3) == size(im_fail, 3)))) % channels

                        % Update NNdb for failed images
                        nndb_fail.add_data(im_fail);
                        nndb_fail.update_attr(is_new_class_for_fail);                        
                        is_new_class_for_fail = false;
                        fail_cls_lbl = cat(2, fail_cls_lbl, uint16([cls_i]));                            
                    else
                        warning('nndb_fail: Failed images are in different resolutions, ignored...')
                    end                                               
                    continue;
                end

                % Filter with pose
                if (~FaceAligner.pose_filter__(pose, info))
                    continue;
                end 

                % Perform alignment with landmarks for img
                img_aligned = FaceAligner.align_face__(img, lmarks_img, desired_face);

                % Save the aligned image
                if (~isempty(save_path))
                    imwrite(img_aligned, fullfile(save_path, cls_name, [num2str(cls_img_i) '_aligned.jpg']), 'jpg');
                end

                % Update NNdb
                nndb_aln.add_data(img_aligned);
                nndb_aln.update_attr(is_new_class);                        
                is_new_class = false;                                            
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function success = pose_filter__(pose, info)
            pose_pitch = [];
            pose_yaw = [];
            pose_roll = [];
            success = true;
            
            if (~isfield(info,'filter')); info.filter = []; end
            if (isfield(info.filter,'pose_pitch')); pose_pitch = info.filter.pose_pitch; end
            if (isfield(info.filter,'pose_yaw')); pose_yaw = info.filter.pose_yaw; end
            if (isfield(info.filter,'pose_roll')); pose_roll = info.filter.pose_roll; end
            
            % pitch check
            if (~isempty(pose_pitch) && (pose.angle(1) < pose_pitch(1) || pose.angle(1) > pose_pitch(2))) 
                success = false;
            end
            
            % yaw check
            if (success && ~isempty(pose_yaw) && (pose.angle(2) < pose_yaw(1) || pose.angle(2) > pose_yaw(2)))
                success = false;
            end
            
            % roll check
            if (success && ~isempty(pose_roll) && (pose.angle(3) < pose_roll(1) || pose.angle(3) > pose_roll(2)))
                success = false;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [img_aligned] = align_face__(img, landmarks, desired_face)
            % Imports
            import nnf.alg.FaceAligner;
            
            leye = mean([landmarks(20,:) ; landmarks(23,:)] , 1); % Left eye centre
            reye = mean([landmarks(26,:) ; landmarks(29,:)] , 1); % Right eye centre  
            mouth = mean([landmarks(32,:) ; landmarks(38,:)] , 1); % average of 2 mouth corners.

            % I use 300 x 300 here, you can change it anytime.
            [img_aligned, ~, ~, ~, ~, ~, ~] = ...
                FaceAligner.autoimalign__(img,desired_face,leye(1), leye(2), reye(1), reye(2), mouth(1), mouth(2));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [aimg alex aley arex arey amx amy] = autoimalign__(img,desired_face,lex,ley,rex,rey,mx,my)
            % IMALIGN align a face image
            %   [AIMG ALEX ALEY AREX AREY AMX AMY]=IMALIGN(IMG,IMHEIGHT,IMWIDTH,LEX,LEY,REX,REY,MX,MY)
            %   aligns a face image.
            %   The aligned image will be plotted in a new window with the eyes and mouth labeled.
            %
            % PARAMETERS
            %
            %   DESIRED_FACE: properties
            %   -----------------------------------
            %   desired_face.height = 100;       % imheight after alignment and crop
            %   desired_face.width = 100;        % imwidth after alignment and crop
            %   desired_face.lex_ratio = 0.25;   % left eye centre `x` position ratio w.r.t  
            %                                    % top, left; calculated from desired width.
            %   desired_face.rex_ratio = 0.75;   % right eye centre `x` position ratio w.r.t
            %                                    % top, left; calculated from desired width.
            %   desired_face.ey_ratio = 0.33;    % both eyes centre `y` position ratio w.r.t
            %                                    % top, left; calculated from desired height.
            %   desired_face.my_ratio = 0.63;    % mouth center `y` position ratio w.r.t
            %                                    % top, left; calculated from desired height.
            %
            %   IMHEIGHT, IMWIDTH: the target height and the width of the aligned image
            %   LEX,LEY: the coordinates of the left eye in the source images
            %   REX,REY: the coordinates of the right eye in the source images
            %   MX,MY: the coordinates of the mouth in the source images
            %
            %   LEYEX, REYEX, EYEY: the target coordinates of the eyes in the aligned images
            %   MOUTHY: the coordinates of the mouth in the aligned images,
            %   mouthx=(reyex+leyex)/2
            %
            %   LEYEX=IMWIDTH/4, REYE=IMWIDTH*3/4, EYEY=IMHEIGHT/3,
            %   MOUTHY=IMHEIGHT*3/4 
            %
            %   The images are aligned by rotation, scale and crop so that the sum
            %   square error of the features to the target coordinates are minimized.
            %
            % RETURNS
            %
            %   AIMG: the aligned image
            %   ALEX, ALEY, AREX, AREY, AMX, AMY: feature coordinates after alignment            
            
            % leyex=imwidth/4;
            % reyex=imwidth*3/4;
            % eyey=imheight/3;
            % mouthy=imheight*3/4;
            % mouthy=imheight*0.63; % preserve chin
            % mouthx=(leyex+reyex)/2;
            
            % Initialize variables
            imwidth = desired_face.width;
            imheight = desired_face.height;                
            lex_ratio = desired_face.lex_ratio;
            rex_ratio = desired_face.rex_ratio;
            ey_ratio = desired_face.ey_ratio;
            my_ratio = desired_face.my_ratio;
            
            % Calculate from ratios
            leyex = imwidth * lex_ratio;
            reyex = imwidth * rex_ratio;
            eyey = imheight * ey_ratio;
            mouthy = imheight * my_ratio;
            mouthx = (leyex+reyex)/2;

            os=size(img);

            % figure;imshow(img);
            % hold on;
            % plot([lex rex mx],[ley rey my],'r+');

            X=[lex rex mx];
            Y=[ley rey my];
            Xc=[leyex reyex mouthx];
            Yc=[eyey eyey mouthy];

            A1=X*X'+Y*Y';
            A2=lex+rex+mx;
            A3=ley+rey+my;
            A4=X*Xc'+Y*Yc';
            A5=X*Yc'-Y*Xc';
            A6=leyex+reyex+mouthx;
            A7=eyey+eyey+mouthy;

            %At=b, t=A\b
            A=[A1 0 A2 A3;0 A1 -A3 A2;A2 -A3 3 0;A3 A2 0 3];
            b=[A4 A5 A6 A7]';
            t=A\b;
            k1=t(1,1);
            k2=t(2,1);
            t1=t(3,1);
            t2=t(4,1);

            rot=atan2(k2,k1);
            scale=(k1*k1+k2*k2)^0.5;
            ox=-t1;
            oy=-t2;
            % we want to rotate rot, crop at ox,oy and then scale 

            %read the file, align, write to target folder
            % rotation
            angle=-rot*180/pi;
           
            % Make output image `aimg` large enough to contain the
            % entire rotated image.
            aimg=imrotate(img,angle,'bilinear');
            
            % after rotation, the (0,0) is changed; 
            % calculate the shifts to adjust the crop
            sx=0;sy=0;
            if 0<angle && angle<=90
                sy=os(2)*sin(-rot);
            end
            if -90<=angle && angle<0 
                sx=os(1)*sin(rot); 
            end
            if 90<=angle && angle<=180
                sx=os(2)*sin(-rot-pi/2); 
                sy=size(aimg,1);
            end
            if -90>=angle && angle>=-180
                sx=size(aimg,2);
                sy=os(1)*sin(rot-pi/2);
            end
            sx=sx*scale;
            sy=sy*scale;

            % scale
            aimg=imresize(aimg,scale,'bilinear'); %, 'OutputSize', [250 250]);
            
            % crop
            cx=round(ox+sx+1);
            cy=round(oy+sy+1);
            aimg=imcrop(aimg,[cx,cy,imwidth-1,imheight-1]); 
            %if img not big enough, pad zeros
            if size(aimg,1)<imheight || size(aimg,2)<imwidth
                timg=cast(zeros(imheight,imwidth,size(aimg,3)),class(aimg));
                if cx<1 
                    px=-cx+1; 
                else
                    px=1; 
                end
                if cy<1 
                    py=-cy+1;
                else
                    py=1;
                end
                timg(py:py+size(aimg,1)-1,px:px+size(aimg,2)-1,:)=aimg;
                aimg=timg;
            end

            % new feature coordinates after alingment
            alex=(lex*cos(rot)-ley*sin(rot))*scale-ox;
            aley=(lex*sin(rot)+ley*cos(rot))*scale-oy;
            arex=(rex*cos(rot)-rey*sin(rot))*scale-ox;
            arey=(rex*sin(rot)+rey*cos(rot))*scale-oy;
            amx=(mx*cos(rot)-my*sin(rot))*scale-ox;
            amy=(mx*sin(rot)+my*cos(rot))*scale-oy;

            % figure;
            % imshow(aimg);
            % hold on;
            % plot([alex arex amx],[aley arey amy],'r+');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

