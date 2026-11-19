function pe = permutation_entropy(y, m, t)
%PERMUTATION_ENTROPY  归一化排列熵（Bandt & Pompe 2002），向量化实现
if nargin < 2, m = 3; end
if nargin < 3, t = 1; end
y = y(:);
n = numel(y) - t*(m-1);
idx = (1:n).' + (0:m-1) * t;
[~, ord] = sort(y(idx), 2);
code = ord * (m .^ (0:m-1)).';       % 每种排列唯一编码
[~, ~, g] = unique(code);
c = accumarray(g, 1);
p = c / sum(c);
pe = -sum(p .* log(p)) / log(factorial(m));
end
