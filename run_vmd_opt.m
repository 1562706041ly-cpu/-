%RUN_VMD_OPT  模块一：用 TTAO / GLO / 随机搜索 优化 VMD 参数并分解信号
%   替代原 main.m 与 main1.m。所有参数在 config.m 中修改。
clear; close all; clc;
setup_paths();
cfg = config();
rng(cfg.seed);
c = cfg.vmd;
out = new_run_dir(cfg, ['vmd_' lower(c.optimizer)]);

[x, fs] = load_signal(c.signal_file, c.fs);
fprintf('信号长度 %d，采样率 %g Hz\n', numel(x), fs);

keyfun = @(z) vmd_key(z, c);
counter = EvalCounter(@(z) vmd_fitness(z, x, c), keyfun);
res = run_optimizer(c.optimizer, counter, 2, c.pop_size, c.max_iter, struct('ttao_step', c.ttao_step));

alpha = res.info.alpha; K = res.info.K;
fprintf('[%s] 最优 alpha = %d, K = %d, 适应度 = %.4f (调用 %d 次，实际计算 %d 次，用时 %.1f s)\n', ...
    res.name, alpha, K, res.best_fit, res.n_calls, res.n_unique, res.time_s);

% 用与适应度完全相同的参数进行最终分解
[u, omega] = vmd_run(x, alpha, K, c);

fig1 = figure('Color', 'w');
plot(res.trace, 'LineWidth', 1.5); grid on;
xlabel('适应度评估次数'); ylabel('当前最优适应度');
title(sprintf('%s 优化 VMD 收敛曲线', res.name));
save_fig(fig1, out, 'convergence', cfg);
fig2 = plot_imfs(u, x, fs, sprintf('%s-VMD (alpha=%d, K=%d)', res.name, alpha, K));
save_fig(fig2, out, 'imfs', cfg);

csvwrite(fullfile(out, 'imfs.csv'), u);
save(fullfile(out, 'result.mat'), 'res', 'u', 'omega', 'alpha', 'K', 'cfg', '-v7');
fprintf('结果已保存到 %s\n', out);

