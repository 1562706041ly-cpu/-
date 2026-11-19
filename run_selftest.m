%RUN_SELFTEST  快速自检（约 1 分钟，不需要任何工具箱）
%   修改代码后先跑一遍，确认核心模块输出正确。
clear; clc;
setup_paths();
cfg = config(); c = cfg.vmd;
n_fail = 0;
check = @(ok, msg) fprintf('[%s] %s\n', ternary(ok, ' OK ', 'FAIL'), msg);

% 1. VMD：两个正弦 + 噪声，应分出两个中心频率
rng(1);
t = (0:999) / 1000;
x = sin(2*pi*50*t) + 0.5*sin(2*pi*200*t) + 0.05*randn(size(t));
[u, om] = vmd_fast(x, 2000, 0, 2, 0, 1, 1e-7, 500);
ok = abs(om(1)*1000 - 50) < 3 && abs(om(2)*1000 - 200) < 3;
check(ok, sprintf('VMD 中心频率 %.1f Hz / %.1f Hz（应为 50 / 200）', om*1000)); n_fail = n_fail + ~ok;
ok = norm(sum(u, 1) - x) / norm(x) < 0.1;
check(ok, 'VMD 各模态之和可重构原信号'); n_fail = n_fail + ~ok;

% 2. 参数编解码一致
ok = true;
for K = c.K_range(1):c.K_range(2)
    [a, k] = vmd_decode(vmd_encode(1234, K, c), c); ok = ok && a == 1234 && k == K;
end
check(ok, 'vmd_encode / vmd_decode 互逆，K 取遍整个范围'); n_fail = n_fail + ~ok;

% 3. 熵指标
ok = permutation_entropy(randn(1, 5000)) > 0.98 && permutation_entropy(1:1000) < 1e-9;
check(ok, '排列熵：白噪声≈1，单调序列=0'); n_fail = n_fail + ~ok;
ok = sample_entropy(sin(2*pi*(1:600)/50)) < sample_entropy(randn(1, 600));
check(ok, '样本熵：正弦 < 白噪声'); n_fail = n_fail + ~ok;

% 4. 优化器：在 2 维二次函数上能接近最优
for name = {'TTAO', 'GLO', 'BKA'}
    cnt = EvalCounter(@(z) deal(sum((z - 0.3).^2), []), @(z) sprintf('%.6f_%.6f', z(1), z(2)), 600);
    r = run_optimizer(name{1}, cnt, 2, 12, 1000);
    ok = r.best_fit < 1e-3;
    check(ok, sprintf('%s 优化二次函数，最优值 %.2e（%d 次评估）', name{1}, r.best_fit, r.n_calls)); n_fail = n_fail + ~ok;
end

% 5. 样本切分与数据划分无交叉
D = make_windows((1:5000).', 2048, 1000, 3);
ok = isequal(D(:, 1).', [1 1001 2001]);
check(ok, '滑窗切分起点正确'); n_fail = n_fail + ~ok;
y = repelem((1:5).', 200);
[a, b, d] = split_indices(y, [0.6 0.2 0.2], 2);
ok = isempty(intersect(a, d)) && isempty(intersect(b, d)) && min(d(y(d) == 1)) - max(b(y(b) == 1)) >= 3;
check(ok, '训练/验证/测试无交叉，边界留空'); n_fail = n_fail + ~ok;

% 6. LSSVM：两团高斯点应几乎全对
X = [randn(100, 2); randn(100, 2) + 4; randn(100, 2) + [4 -4]]; yy = repelem((1:3).', 100);
p = lssvm_ovo(X(1:2:end, :), yy(1:2:end), X(2:2:end, :), 10, 1, 3);
ok = mean(p == yy(2:2:end)) > 0.95;
check(ok, sprintf('LSSVM 一对一分类准确率 %.1f%%', 100 * mean(p == yy(2:2:end)))); n_fail = n_fail + ~ok;

% 7. 同步提取变换：单频信号能量集中在对应频率
x = cos(2*pi*0.1*(0:1023));
[Te, f] = set_transform(x);
[~, i] = max(sum(abs(Te), 2));
ok = abs(f(i) - 0.1) < 0.01;
check(ok, sprintf('同步提取变换峰值频率 %.3f（应为 0.100）', f(i))); n_fail = n_fail + ~ok;

% 8. 面积平均缩放：常数矩阵不变，总能量守恒
A = zeros(320, 2048); A(1:7:end, 3:11:end) = 1;
B = resize_area(A, [64 64]);
ok = abs(mean(B(:)) - mean(A(:))) < 1e-12 && max(max(abs(resize_area(ones(320, 2048), [64 64]) - 1))) < 1e-12;
check(ok, '面积平均缩放保持均值（稀疏矩阵能量不丢失）'); n_fail = n_fail + ~ok;

fprintf('\n自检完成：%d 项失败\n', n_fail);
