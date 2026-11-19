%RUN_VMD_COMPARE  优化器对比实验（替代原 comparison.m）
%   - 每个优化器独立运行 cfg.cmp.n_runs 次，统一适应度评估预算 cfg.cmp.max_evals；
%   - 报告 最优/均值/标准差/最差、命中网格最优的比例，以及 Wilcoxon 秩和检验 p 值；
%   - 网格穷举给出参考最优，判断优化器是否真的找到了好解。
clear; close all; clc;
setup_paths();
cfg = config();
c = cfg.vmd; m = cfg.cmp;
out = new_run_dir(cfg, 'vmd_compare');
[x, fs] = load_signal(c.signal_file, c.fs);

% 共享缓存：同一参数只做一次 VMD（不影响各算法的计数与公平性）
shared = containers.Map('KeyType', 'char', 'ValueType', 'any');
fit_fun = @(z) cached_fitness(z, x, c, shared);
keyfun  = @(z) vmd_key(z, c);

%% 网格穷举参考
grid_best = NaN;
if m.do_grid
    alphas = c.alpha_range(1):100:c.alpha_range(2);
    Ks = c.K_range(1):c.K_range(2);
    G = zeros(numel(Ks), numel(alphas));
    for i = 1:numel(Ks)
        for j = 1:numel(alphas)
            G(i, j) = fit_fun(vmd_encode(alphas(j), Ks(i), c));
        end
    end
    [grid_best, gi] = min(G(:)); [ki, ai] = ind2sub(size(G), gi);
    fprintf('网格参考最优：alpha=%d, K=%d, 适应度=%.4f\n', alphas(ai), Ks(ki), grid_best);
    fig = figure('Color', 'w'); imagesc(alphas, Ks, G); axis xy; colorbar;
    xlabel('\alpha'); ylabel('K'); title('适应度地形（网格穷举）');
    save_fig(fig, out, 'landscape', cfg);
end

%% 多次独立运行
names = m.optimizers; nO = numel(names);
best = zeros(m.n_runs, nO); params = zeros(m.n_runs, 2, nO);
traces = nan(m.max_evals, m.n_runs, nO);
for o = 1:nO
    for r = 1:m.n_runs
        rng(cfg.seed + r);                    % 各算法使用相同的种子序列
        counter = EvalCounter(fit_fun, keyfun, m.max_evals);
        res = run_optimizer(names{o}, counter, 2, c.pop_size, 10000, struct('ttao_step', c.ttao_step));
        best(r, o) = res.best_fit;
        params(r, :, o) = [res.info.alpha res.info.K];
        tr = res.trace; traces(1:numel(tr), r, o) = tr;
        if numel(tr) < m.max_evals, traces(numel(tr)+1:end, r, o) = tr(end); end
    end
    fprintf('%-7s 完成 %d 次运行\n', names{o}, m.n_runs);
end

%% 统计表
tol = 1e-3;                          % 与网格最优相差 < 0.001 视为命中（网格 alpha 步长为 100）
header = {'算法', '最优', '均值', '标准差', '最差', '命中网格最优比例', 'K众数', '相对第一个算法p值'};
rows = cell(nO, numel(header));
fprintf('\n%-8s %8s %8s %8s %8s %8s %5s %8s\n', '算法', '最优', '均值', '标准差', '最差', '命中率', 'K众数', 'p值');
for o = 1:nO
    b = best(:, o);
    hit = mean(b <= grid_best + tol);
    p = rank_sum_test(best(:, 1), b);
    rows(o, :) = {names{o}, min(b), mean(b), std(b), max(b), hit, mode(params(:, 2, o)), p};
    fprintf('%-8s %8.4f %8.4f %8.4f %8.4f %7.0f%% %5d %8.3g\n', names{o}, min(b), mean(b), std(b), max(b), 100*hit, mode(params(:, 2, o)), p);
end
write_table_csv(fullfile(out, 'compare_stats.csv'), header, rows);

fig = figure('Color', 'w'); hold on; grid on;
for o = 1:nO, plot(mean(traces(:, :, o), 2), 'LineWidth', 1.5); end
if ~isnan(grid_best), plot([1 m.max_evals], [grid_best grid_best], 'k--'); end
legend([names, {'网格最优'}]); xlabel('适应度评估次数'); ylabel(sprintf('平均最优适应度（%d 次）', m.n_runs));
title('优化器收敛对比（相同评估预算）');
save_fig(fig, out, 'convergence_compare', cfg);
fig = figure('Color', 'w'); hold on; grid on;
for o = 1:nO   % 散点+均值图，不依赖统计工具箱的 boxplot
    plot(o + 0.08*randn(m.n_runs, 1), best(:, o), 'o');
    plot(o + [-0.25 0.25], mean(best(:, o)) * [1 1], 'k-', 'LineWidth', 2);
end
set(gca, 'XTick', 1:nO, 'XTickLabel', names); xlim([0.5 nO+0.5]);
ylabel('最终适应度'); title('多次运行结果分布（横线为均值）');
save_fig(fig, out, 'distribution', cfg);
save(fullfile(out, 'compare.mat'), 'best', 'params', 'traces', 'grid_best', 'names', 'cfg', '-v7');
fprintf('结果已保存到 %s\n', out);

