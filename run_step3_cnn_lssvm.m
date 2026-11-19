%RUN_STEP3_CNN_LSSVM  模块四：深度网络特征提取 + BKA 优化 LSSVM 分类（替代 CNN_BKA_LSSVM.m）
%   需要 MATLAB Deep Learning Toolbox（R2021a 及以上）。
%   v2.2：网络由 cfg.cls.model 选择（'cnn' | 'resnet' | 'eca_resnet'，见 src/build_net.m），默认 'resnet'；
%         可用 cfg.cls.cross_file 指定第二个测试集（如 30-2 工况图库）做跨工况测试。
%   主要改动：
%   1) 训练/验证/测试按时间顺序划分且边界留空，测试集只在最后评估一次；
%   2) CNN 用验证集早停；LSSVM 超参数由 BKA 在“训练+验证集 k 折交叉验证”上寻优，
%      不再用测试集准确率当适应度（原代码的数据泄露问题）；
%   3) 送入 LSSVM 的是 fc1 层 64 维特征（原代码用 5 维输出层，LSSVM 作用很小）；
%   4) 重复 n_repeats 次，报告均值±标准差，并输出精确率/召回率/F1、混淆矩阵。
clear; close all; clc;
if strcmpi(config().cls.lssvm_impl, 'lssvmlab'), setup_paths('lssvm'); else, setup_paths(); end
cfg = config(); c = cfg.cls;
out = new_run_dir(cfg, 'cnn_bka_lssvm');

[X, y] = load_tfr_images(c.image_file);
has_cross = isfield(c, 'cross_file') && ~isempty(c.cross_file);
if has_cross, [Xc, yc] = load_tfr_images(c.cross_file); end
nC = numel(unique(y));
if numel(cfg.smp.class_names) == nC
    names = cfg.smp.class_names;
else
    names = arrayfun(@(k) sprintf('C%d', k), 1:nC, 'UniformOutput', false);
end
[itr, iva, ite] = split_indices(y, c.split, c.gap);
idev = [itr; iva];
fprintf('训练 %d / 验证 %d / 测试 %d\n', numel(itr), numel(iva), numel(ite));

layers = build_net(c.model, size(X, 1:3), nC);
fprintf('网络：%s\n', c.model);

R = c.n_repeats;
acc_cnn = zeros(R, 1); acc_lssvm = zeros(R, 1); f1_lssvm = zeros(R, 1);
best_params = zeros(R, 2); n_invalid = zeros(R, 1);
acc_cnn_cross = nan(R, 1); acc_lssvm_cross = nan(R, 1);
for r = 1:R
    rng(cfg.seed + r);
    opts = trainingOptions('adam', ...
        'MaxEpochs', c.max_epochs, 'MiniBatchSize', c.batch, ...
        'InitialLearnRate', c.lr, 'Shuffle', 'every-epoch', ...
        'ValidationData', {X(:, :, :, iva), categorical(y(iva))}, ...
        'ValidationFrequency', max(1, floor(numel(itr) / c.batch)), ...
        'ValidationPatience', c.patience, 'OutputNetwork', 'best-validation-loss', ...
        'GradientThreshold', 1, 'ExecutionEnvironment', c.exec_env, ...
        'Verbose', false, 'Plots', 'none');
    net = trainNetwork(X(:, :, :, itr), categorical(y(itr)), layers, opts);
    p_cnn = double(classify(net, X(:, :, :, ite)));
    acc_cnn(r) = mean(p_cnn == y(ite));

    F = double(activations(net, X, c.feat_layer, 'OutputAs', 'rows'));
    st = lssvm_stage(F(idev, :), y(idev), F(ite, :), y(ite), c, nC);
    M = st.metrics;
    acc_lssvm(r) = M.accuracy; f1_lssvm(r) = M.macro_f1;
    best_params(r, :) = [st.gam st.sig2]; n_invalid(r) = st.n_invalid;
    if has_cross                       % 跨工况：同一个模型、同一组 LSSVM 超参数，直接预测
        acc_cnn_cross(r) = mean(double(classify(net, Xc)) == yc);
        Fc = double(activations(net, Xc, c.feat_layer, 'OutputAs', 'rows'));
        Fd = (F(idev, :) - st.mu) ./ st.sd;
        pc = lssvm_ovo(Fd, y(idev), (Fc - st.mu) ./ st.sd, st.gam, st.sig2, nC);
        acc_lssvm_cross(r) = mean(pc == yc);
        fprintf('        跨工况：CNN %.2f%% | +BKA-LSSVM %.2f%%\n', 100*acc_cnn_cross(r), 100*acc_lssvm_cross(r));
    end
    fprintf('第 %d 次：CNN %.2f%% | CNN-BKA-LSSVM %.2f%%（macro-F1 %.4f，gam=%.3g，sig2=%.3g，CV 错误率 %.4f）\n', ...
        r, 100*acc_cnn(r), 100*M.accuracy, M.macro_f1, st.gam, st.sig2, st.cv_error);
    if r == 1
        fig = figure('Color', 'w'); plot(st.trace, 'LineWidth', 1.5); grid on;
        xlabel('适应度评估次数'); ylabel('交叉验证错误率'); title('BKA 优化 LSSVM 收敛曲线');
        save_fig(fig, out, 'bka_convergence', cfg);
        fig = plot_confusion(M.confusion, names, sprintf('测试集混淆矩阵 (准确率 %.2f%%)', 100*M.accuracy));
        save_fig(fig, out, 'confusion_test', cfg);
        if exist('tsne', 'file')            % t-SNE 画测试集特征（需统计工具箱）
            E2 = tsne((F(ite, :) - st.mu) ./ st.sd);
            fig = figure('Color', 'w'); gscatter(E2(:, 1), E2(:, 2), categorical(names(y(ite)).'));
            title('测试集 fc1 特征 t-SNE'); save_fig(fig, out, 'tsne_test', cfg);
        end
        gam = st.gam; sig2 = st.sig2; mu = st.mu; sd = st.sd; %#ok<NASGU>
        save(fullfile(out, 'model_run1.mat'), 'net', 'gam', 'sig2', 'mu', 'sd', 'M', '-v7');
    end
end

fprintf('\n====== %d 次重复实验（测试集）======\n', R);
fprintf('网络 softmax   : %.2f%% ± %.2f%%\n', 100*mean(acc_cnn), 100*std(acc_cnn));
fprintf('+BKA-LSSVM     : %.2f%% ± %.2f%%，macro-F1 %.4f ± %.4f\n', ...
    100*mean(acc_lssvm), 100*std(acc_lssvm), mean(f1_lssvm), std(f1_lssvm));
fprintf('无法判别样本数 : %d\n', sum(n_invalid));
if has_cross
    fprintf('跨工况 网络    : %.2f%% ± %.2f%%\n', 100*mean(acc_cnn_cross), 100*std(acc_cnn_cross));
    fprintf('跨工况 +LSSVM  : %.2f%% ± %.2f%%\n', 100*mean(acc_lssvm_cross), 100*std(acc_lssvm_cross));
end
write_table_csv(fullfile(out, 'repeats.csv'), {'run', 'acc_cnn', 'acc_lssvm', 'macro_f1', 'gam', 'sig2', 'invalid', 'acc_cnn_cross', 'acc_lssvm_cross'}, ...
    num2cell([(1:R).' acc_cnn acc_lssvm f1_lssvm best_params n_invalid acc_cnn_cross acc_lssvm_cross]));
save(fullfile(out, 'summary.mat'), 'acc_cnn', 'acc_lssvm', 'acc_cnn_cross', 'acc_lssvm_cross', 'f1_lssvm', 'best_params', 'cfg', '-v7');
fprintf('结果已保存到 %s\n', out);
