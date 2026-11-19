function folds = stratified_folds(y, k)
%STRATIFIED_FOLDS  分层 k 折编号（每类样本均匀分到各折）
y = y(:); folds = zeros(size(y));
for c = unique(y).'
    idx = find(y == c);
    idx = idx(randperm(numel(idx)));
    folds(idx) = mod(0:numel(idx)-1, k).' + 1;
end
end
