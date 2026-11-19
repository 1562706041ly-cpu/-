function e = info_entropy(x, nbins)
%INFO_ENTROPY  基于幅值直方图的信息熵（替代 Image Toolbox 的 entropy）
if nargin < 2, nbins = 256; end
x = x(:);
rng_x = max(x) - min(x);
if rng_x == 0, e = 0; return; end
edges = linspace(min(x), max(x), nbins + 1);
c = histc(x, edges); c(end-1) = c(end-1) + c(end); c = c(1:end-1);
p = c(c > 0) / numel(x);
e = -sum(p .* log2(p));
end
