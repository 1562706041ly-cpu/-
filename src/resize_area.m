function B = resize_area(A, sz)
%RESIZE_AREA  按面积平均缩小矩阵（抗混叠，不依赖图像处理工具箱）
%   输出的每个像素等于它覆盖的原矩阵区域的加权平均。
%   同步提取变换的时频矩阵很稀疏（约 7% 非零），用 resize_bilinear 逐点取样
%   从 2048 列缩到 64 列时只取到约 1/32 的列，大部分能量被丢掉或混叠；
%   面积平均保留了全部能量。
[h, w] = size(A);
B = area_weights(sz(1), h) * A * area_weights(sz(2), w).';
end

function W = area_weights(m, n)
% m x n 权重矩阵：第 i 行是输出像素 i 在原坐标 [(i-1)*n/m, i*n/m] 上的覆盖比例
W = zeros(m, n);
s = n / m;
for i = 1:m
    a = (i - 1) * s; b = i * s;
    for j = floor(a) + 1:min(ceil(b), n)
        W(i, j) = min(b, j) - max(a, j - 1);
    end
end
W = W ./ sum(W, 2);
end
