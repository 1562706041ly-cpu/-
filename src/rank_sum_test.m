function p = rank_sum_test(a, b)
%RANK_SUM_TEST  Wilcoxon 秩和检验（正态近似、含结校正），双侧 p 值
%   不依赖统计工具箱；样本量 >= 10 时近似效果良好。
a = a(:); b = b(:);
n1 = numel(a); n2 = numel(b); n = n1 + n2;
[s, ~] = sort([a; b]);
r = tied_rank([a; b]);
W = sum(r(1:n1));
mu = n1 * (n + 1) / 2;
[~, ~, g] = unique(s); tcount = accumarray(g, 1);
sigma2 = n1 * n2 / 12 * ((n + 1) - sum(tcount.^3 - tcount) / (n * (n - 1)));
if sigma2 <= 0, p = 1; return; end
z = (W - mu - 0.5 * sign(W - mu)) / sqrt(sigma2);
p = erfc(abs(z) / sqrt(2));
end

function r = tied_rank(x)
[xs, idx] = sort(x);
n = numel(x); r = zeros(n, 1); i = 1;
while i <= n
    j = i;
    while j < n && xs(j+1) == xs(i), j = j + 1; end
    r(idx(i:j)) = (i + j) / 2;
    i = j + 1;
end
end
