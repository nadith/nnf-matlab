classdef KMEANS
    % KMEANS: K-means clustering algorithm
    %   Refer method specific help for more details. 
    %
    %   Currently Support:
    %   ------------------
    %   - LDA.do (L2 distance)
    
    % Copyright @ Qilin Li, Curtin University (li.qilin@postgrad.curtin.edu.au).
    
    properties        
    end
    
    methods (Access = public, Static)
    	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Interface
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [label, centers] = do(nndb, info)
            % do Kmeans clustering
            %
            % Parameters
            % ----------
            % nndb : nnf.db.NNdb
            %     Data object that contains the database.
            %
            % info : struct, optional
            %     Provide additional information to perform Kmeans. (Default value = []).
            %
            %
            % Returns
            % -------
            % label : nndb.n * 1 vector, indicating the labels for each
            %     sample
            %
            % centers : nndb.n * nndb.features vector. Each row is a
            %     centes of a class
            %
            % Examples
            % --------
            % import nnf.alg.KMEANS;
            % info.Replicates = 10;
            % info.MaxIter = 1000;
            % [label, centers] = KMEANS.do(nndb, info)
            %
            
            % Imports
            import nnf.libs.DengCai;
            
            % Set defaults for arguments
            if (nargin < 2), info = []; end
            if (~isfield(info,'MaxIter')); info.MaxIter = 50; end
            if (~isfield(info,'Replicates')); info.Replicates = 1; end
            
            % Perform kmeans
            [label, centers] = DengCai.litekmeans(nndb.features', nndb.cls_n, 'MaxIter', info.MaxIter, ...
                                                  'Replicates', info.Replicates);
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    methods (Access = private, Static)
    end    
end

