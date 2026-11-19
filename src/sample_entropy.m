function se = sample_entropy(x, m, r)
%SAMPLE_ENTROPY  样本熵（Richman & Moorman 2000）
if nargin < 2, m = 2; end
x = x(:);
if nargin < 3, r = 0.2 * std(x); end
N = numel(x);
B = count_match(x, m, r, N);
A = count_match(x, m + 1, r, N);
if A == 0 || B == 0
    se = Inf;
else
    se = -log(A / B);
end
end

function c = count_match(x, m, r, N)
M = N - m;                          % 两个长度下使用相同数量的模板
X = zeros(M, m);
for k = 1:m, X(:, k) = x(k:k+M-1); end
c = 0;
for i = 1:M-1
    d = max(abs(X(i+1:end, :) - X(i, :)), [], 2);
    c = c + sum(d <= r);
end
end
