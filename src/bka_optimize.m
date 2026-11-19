function [best_fit, best_pos, curve] = bka_optimize(pop, T, lb, ub, dim, fobj)
%BKA_OPTIMIZE  黑翅鸢优化算法（Wang et al., Artif Intell Rev 2024）自研实现
%   相对网上流传版本的修正：
%   1) 随机数 r 每次迭代、每个个体重新生成（原版只在开头生成一次）；
%   2) 柯西变异对每一维独立采样（原版只用最后一维的值）；
%   3) 每代结束后正确地从全种群取最优（原版用向量与标量比较且取了最后一个个体）。
p = 0.9;
lb = lb(:).'; ub = ub(:).';
X = lb + rand(pop, dim) .* (ub - lb);
F = zeros(pop, 1);
for i = 1:pop, F(i) = fobj(X(i, :)); end
[best_fit, ib] = min(F); best_pos = X(ib, :);
curve = zeros(1, T);
for t = 1:T
    n = 0.05 * exp(-2 * (t/T)^2);
    for i = 1:pop
        % ---- 攻击行为 ----
        r = rand;
        if p < r
            Xn = X(i, :) + n * (1 + sin(r)) * X(i, :);
        else
            Xn = X(i, :) .* (n * (2*rand(1, dim) - 1) + 1);
        end
        Xn = min(max(Xn, lb), ub);
        fn = fobj(Xn);
        if fn < F(i), X(i, :) = Xn; F(i) = fn; end
        % ---- 迁徙行为（柯西变异）----
        m = 2 * sin(r + pi/2);
        s = randi(pop);
        c = tan((rand(1, dim) - 0.5) * pi);
        if F(i) < F(s)
            Xn = X(i, :) + c .* (X(i, :) - best_pos);
        else
            Xn = X(i, :) + c .* (best_pos - m .* X(i, :));
        end
        Xn = min(max(Xn, lb), ub);
        fn = fobj(Xn);
        if fn < F(i), X(i, :) = Xn; F(i) = fn; end
        if F(i) < best_fit, best_fit = F(i); best_pos = X(i, :); end
    end
    curve(t) = best_fit;
end
end
