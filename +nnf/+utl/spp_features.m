function nndb_spp = spp_features(nndb, info) 
    % TODO: Comment: Get normalized 2D Feature matrix (double) for specified class labels.
    %
    % Parameters
    % ----------
    % cls_lbl : uint16, optional
    %     featres for class denoted by cls_lbl.
    %
    % norm : string, optional
    %     'l1', 'l2', 'max', normlization for each column. (Default value = []).
    %
    
    % Imports
    import nnf.db.NNdb;
    import nnf.db.Format;
    import nnf.db.DbSlice;
    import nnf.core.generators.NNPatchGenerator;
    
    
    sel = info.sel;
    
    
    gen1 = NNPatchGenerator(nndb.h, nndb.w, 32, 32, 32, 32);
    gen2 = NNPatchGenerator(nndb.h, nndb.w, 16, 16, 16, 16);
    gen3 = NNPatchGenerator(nndb.h, nndb.w, 8, 8, 8, 8);
 
    patch_gens = [gen1 gen2 gen3];
    
    max_pooling = false;
    avg_pooling = true;
    
    features = [];
    for i=1:numel(patch_gens)
             
        patch_gen = patch_gens(i);
        sel.nnpatches = patch_gen.generate_nnpatches();
        [nndbs_tr, ~, ~, ~, ~, ~, ~] = DbSlice.slice(nndb, sel);
        
        % Iterate each patch nndb
        for j=1:numel(nndbs_tr)
            if (max_pooling)
                features = [features; max(nndbs_tr(j).features, [], 1)];
                
            elseif (avg_pooling)
                features = [features; mean(nndbs_tr(j).features, 1)];
                
            end
        end       
    end
    
    nndb_spp = NNdb([nndb.name '- SPP'], features, nndb.n_per_class, false, nndb.cls_lbl, Format.H_N);
end