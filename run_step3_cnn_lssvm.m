%RUN_STEP3_CNN_LSSVM  模块四：CNN 特征提取 + BKA 优化 LSSVM 分类（替代 CNN_BKA_LSSVM.m）
%   需要 MATLAB Deep Learning Toolbox（R2021a 及以上）。主要改动：
%   1) 训练/验证/测试按时间顺序划分且边界留空，测试集只在最后评估一次；
%   2) CNN 用验证集早停；LSSVM 超参数由 BKA 在“训练+验证集 k 折交叉验证”上寻优，
%      不再用测试集准确率当适应度（原代码的数据泄露问题）；
%   3) 送入 LSSVM 的是 fc1 层 64 维特征（原代码用 5 维输出层，LSSVM 作用很小）；
%   4) 重复 n_repeats 次，报告均值±标准差，并输出精确率/召回率/F1、混淆矩阵。
clear; close all; clc;
if strcmpi(config().cls.lssvm_impl, 'lssvmlab'), setup_paths('lssvm'); else, setup_paths(); end
cfg = config(); c = cfg.cls;
out = new_run_dir(cfg, 'cnn_bka_lssvm');

S = load(c.image_file);
imgs = S.resizeimg(1, :); y = cell2mat(S.resizeimg(2, :)).';
X = cat(4, imgs{:});
nC = numel(unique(y));
if numel(cfg.smp.class_names) == nC
    names = cfg.smp.class_names;
else
    names = arrayfun(@(k) sprintf('C%d', k), 1:nC, 'UniformOutput', false);
end
[itr, iva, ite] = split_indices(y, c.split, c.gap);
idev = [itr; iva];
fprintf('训练 %d / 验证 %d / 测试 %d\n', numel(itr), numel(iva), numel(ite));

layers = [
    imageInputLayer(size(X, 1:3), 'Name', 'input', 'Normalization', 'zscore')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    globalAveragePooling2dLayer('Name', 'gap')
    dropoutLayer(0.3, 'Name', 'drop')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    fullyConnectedLayer(nC, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

R = c.n_repeats;
acc_cnn = zeros(R, 1); acc_lssvm = zeros(R, 1); f1_lssvm = zeros(R, 1);
best_params = zeros(R, 2); n_invalid = zeros(R, 1);
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
fprintf('CNN (softmax)  : %.2f%% ± %.2f%%\n', 100*mean(acc_cnn), 100*std(acc_cnn));
fprintf('CNN-BKA-LSSVM  : %.2f%% ± %.2f%%，macro-F1 %.4f ± %.4f\n', ...
    100*mean(acc_lssvm), 100*std(acc_lssvm), mean(f1_lssvm), std(f1_lssvm));
fprintf('无法判别样本数 : %d\n', sum(n_invalid));
write_table_csv(fullfile(out, 'repeats.csv'), {'run', 'acc_cnn', 'acc_lssvm', 'macro_f1', 'gam', 'sig2', 'invalid'}, ...
    num2cell([(1:R).' acc_cnn acc_lssvm f1_lssvm best_params n_invalid]));
save(fullfile(out, 'summary.mat'), 'acc_cnn', 'acc_lssvm', 'f1_lssvm', 'best_params', 'cfg', '-v7');
fprintf('结果已保存到 %s\n', out);
