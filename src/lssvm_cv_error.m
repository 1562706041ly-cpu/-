function err = lssvm_cv_error(z, X, y, folds, cfg_cls, n_classes)
%LSSVM_CV_ERROR  LSSVM 超参数适应度：开发集内分层 k 折交叉验证错误率
%   z∈[0,1]^2 映射到对数空间的 gam、sig2。测试集完全不参与。
%   核矩阵对整个开发集只计算一次，各折直接取子矩阵。
[gam, sig2] = lssvm_decode(z, cfg_cls);
use_fast = ~isfield(cfg_cls, 'lssvm_impl') || strcmpi(cfg_cls.lssvm_impl, 'fast');
if use_fast, K = rbf_kernel(X, X, sig2); end
wrong = 0;
for k = 1:max(folds)
    te = folds == k; tr = ~te;
    if use_fast
        p = lssvm_ovo([], y(tr), [], gam, sig2, n_classes, K(tr, tr), K(te, tr));
    else
        p = lssvm_fit_predict(X(tr, :), y(tr), X(te, :), gam, sig2, n_classes);
    end
    wrong = wrong + sum(p(:) ~= y(te));
end
err = wrong / numel(y);
end
