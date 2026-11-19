function [pred, n_invalid] = lssvm_ovo(Xtr, ytr, Xte, gam, sig2, n_classes, Ktr, Kte)
%LSSVM_OVO  自研 LS-SVM 多分类（一对一投票，RBF 核 K = exp(-||x-z||^2 / sig2)）
%   与 LSSVMlab 的 trainlssvm/simlssvm（code_OneVsOne）数学上等价，但：
%   - 全部向量化，核矩阵只算一次并在各二分类器间复用，速度快一个数量级以上；
%   - 不依赖第三方 LSSVMlab；
%   - 可传入预先计算好的核矩阵 Ktr/Kte（交叉验证时复用）。
%   投票平局时取决策值之和最大的类别。
ytr = ytr(:);
if nargin < 7 || isempty(Ktr), Ktr = rbf_kernel(Xtr, Xtr, sig2); end
if nargin < 8 || isempty(Kte), Kte = rbf_kernel(Xte, Xtr, sig2); end
nte = size(Kte, 1);
votes = zeros(nte, n_classes);
score = zeros(nte, n_classes);
for a = 1:n_classes-1
    for b = a+1:n_classes
        idx = find(ytr == a | ytr == b);
        if isempty(idx), continue; end
        yb = 2 * (ytr(idx) == a) - 1;              % a 类 = +1，b 类 = -1
        n = numel(idx);
        Om = (yb * yb.') .* Ktr(idx, idx) + eye(n) / gam;
        A = [0, yb.'; yb, Om];
        sol = A \ [0; ones(n, 1)];
        bias = sol(1); alpha = sol(2:end);
        d = Kte(:, idx) * (alpha .* yb) + bias;
        votes(:, a) = votes(:, a) + (d > 0);
        votes(:, b) = votes(:, b) + (d <= 0);
        score(:, a) = score(:, a) + d;
        score(:, b) = score(:, b) - d;
    end
end
[~, pred] = max(votes + 1e-6 * tanh(score), [], 2);
bad = any(~isfinite(score), 2);
n_invalid = sum(bad);
pred(bad) = 0;
end
