function [tr, val, te] = divideblockex(range, tr_ratio, val_ratio, te_ratio)
    % Divide range into three sets using blocks of indices
    
    r1 = tr_ratio + val_ratio;
    range_0 = range - min(range) + 1;
    r1_en = ceil(max(range_0) * r1);
    r2_count = floor(max(range_0) * te_ratio);
    te = range(r1_en+1:r1_en + r2_count);

    r2_en = r1_en;
    r1_en = ceil((r1_en / r1) * tr_ratio);

    tr = range(1:r1_en);
    val = range(r1_en + 1:r2_en);
end

