function [pred, n_invalid] = lssvm_fit_predict(Xtr, ytr, Xte, gam, sig2, n_classes)
%LSSVM_FIT_PREDICT  LSSVM 多分类（一对一编码，RBF 核）
%   gam/sig2 个数按二分类器数量 nchoosek(n_classes,2) 设置（原代码写死为 45）。
%   无法判别的样本（LSSVMlab 返回 -Inf/NaN）记为 0，计入错误，不再随机赋值。
nb = nchoosek(n_classes, 2);
model = initlssvm(Xtr, ytr(:), 'c', [], [], 'RBF_kernel');
model.code = 'changed';
model.codetype = 'code_OneVsOne';
model.gam = gam * ones(1, nb);
model.kernel_pars = sig2 * ones(1, nb);
model = trainlssvm(model);
pred = simlssvm(model, Xte);
bad = ~isfinite(pred);
n_invalid = sum(bad);
pred(bad) = 0;
end
