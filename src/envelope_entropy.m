function e = envelope_entropy(x)
%ENVELOPE_ENTROPY  包络熵（不依赖信号处理工具箱）
env = abs(analytic_signal(x));
p = env / sum(env);
p = p(p > 0);
e = -sum(p .* log(p));
end
