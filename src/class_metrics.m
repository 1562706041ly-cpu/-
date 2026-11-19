function M = class_metrics(y_true, y_pred, n_classes)
%CLASS_METRICS  准确率、各类精确率/召回率/F1、宏平均 F1、混淆矩阵
y_true = y_true(:); y_pred = y_pred(:);
C = zeros(n_classes);
for i = 1:numel(y_true)
    if y_pred(i) >= 1 && y_pred(i) <= n_classes
        C(y_true(i), y_pred(i)) = C(y_true(i), y_pred(i)) + 1;
    end
end
tp = diag(C);
prec = tp ./ max(sum(C, 1).', 1);
rec  = tp ./ max(sum(C, 2), 1);
f1 = 2 * prec .* rec ./ max(prec + rec, eps);
M = struct('accuracy', mean(y_true == y_pred), 'precision', prec, 'recall', rec, ...
           'f1', f1, 'macro_f1', mean(f1), 'confusion', C);
end
