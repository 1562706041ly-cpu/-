function out = lssvm_stage(Fdev, ydev, Fte, yte, cfg_cls, n_classes)
%LSSVM_STAGE  BKA 优化 LSSVM：开发集内 k 折交叉验证寻优 -> 开发集重训 -> 测试集评估一次
%   Fdev/ydev：训练+验证集特征与标签；Fte/yte：测试集（只在最后使用）
[Fdev, mu, sd] = zscore_fit(Fdev);
Fte = (Fte - mu) ./ sd;
folds = stratified_folds(ydev, cfg_cls.cv_folds);
counter = EvalCounter(@(z) deal(lssvm_cv_error(z, Fdev, ydev, folds, cfg_cls, n_classes), []), ...
                      @(z) sprintf('%.4f_%.4f', z(1), z(2)));
res = run_optimizer('BKA', counter, 2, cfg_cls.bka_pop, cfg_cls.bka_iter);
[gam, sig2] = lssvm_decode(res.best_z, cfg_cls);
if ~isfield(cfg_cls, 'lssvm_impl') || strcmpi(cfg_cls.lssvm_impl, 'fast')
    [pred, n_invalid] = lssvm_ovo(Fdev, ydev, Fte, gam, sig2, n_classes);
else
    [pred, n_invalid] = lssvm_fit_predict(Fdev, ydev, Fte, gam, sig2, n_classes);
end
M = class_metrics(yte, pred, n_classes);
out = struct('gam', gam, 'sig2', sig2, 'cv_error', res.best_fit, 'trace', res.trace, ...
             'pred', pred, 'n_invalid', n_invalid, 'metrics', M, 'mu', mu, 'sd', sd, ...
             'n_calls', res.n_calls);
end
