function [itr, iva, ite] = split_indices(labels, ratios, gap)
%SPLIT_INDICES  每类按时间顺序划分训练/验证/测试，并在边界丢弃 gap 个样本
%   滑窗样本相互重叠（窗长 2048、步长 1000 时相邻 2 个样本重叠），
%   丢弃边界样本可避免训练集和测试集共享同一段原始信号。
labels = labels(:);
itr = []; iva = []; ite = [];
for c = unique(labels).'
    idx = find(labels == c);
    n = numel(idx);
    ntr = floor(ratios(1) * n); nva = floor(ratios(2) * n);
    itr = [itr; idx(1:ntr - gap)];
    iva = [iva; idx(ntr + 1:ntr + nva - gap)];
    ite = [ite; idx(ntr + nva + 1:n)];
end
end
