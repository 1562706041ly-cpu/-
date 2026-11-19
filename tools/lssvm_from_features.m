function lssvm_from_features(prefix, seed, out_csv)
%LSSVM_FROM_FEATURES  读取 py_cnn_features.py 导出的 fc1 特征，跑 BKA-LSSVM 阶段
%   仅用于没有 Deep Learning Toolbox 时的验证；正式流程用 run_step3_cnn_lssvm.m。
%   划分与 CNN 训练时完全相同；测试集只在最后评估一次。
setup_paths();
cfg = config(); c = cfg.cls;
rng(seed);
F = csvread([prefix '_F.csv']);
y = csvread([prefix '_y.csv']);
sp = csvread([prefix '_split.csv']);
idev = sp(sp(:, 2) <= 2, 1); ite = sp(sp(:, 2) == 3, 1);
nC = numel(unique(y));
st = lssvm_stage(F(idev, :), y(idev), F(ite, :), y(ite), c, nC);
acc_cnn = csvread([prefix '_cnnacc.txt']);
fid = fopen(out_csv, 'a');
fprintf(fid, '%s,%d,%.4f,%.4f,%.4f,%.4g,%.4g,%.4f,%d\n', prefix, seed, acc_cnn, ...
        st.metrics.accuracy, st.metrics.macro_f1, st.gam, st.sig2, st.cv_error, st.n_invalid);
fclose(fid);
fprintf('%s seed %d: CNN %.2f%% | CNN-BKA-LSSVM %.2f%% (CV err %.4f, %d evals)\n', prefix, seed, ...
        100*acc_cnn, 100*st.metrics.accuracy, st.cv_error, st.n_calls);
end
