function img = tfr_to_rgb(M, img_size, cmap, method)
%TFR_TO_RGB  时频矩阵 -> 归一化 RGB 图像（直接生成，不再截图保存 JPG）
%   原流程用 figure + saveas 保存 JPG 再读回缩放，速度慢且有压缩损失；
%   这里直接计算幅值、缩放、对数压缩并映射颜色。
%   method：'area'（默认，面积平均，抗混叠）| 'bilinear'（v2.0 的做法，逐点取样）
if nargin < 3 || isempty(cmap), cmap = jet(256); end
if nargin < 4 || isempty(method), method = 'area'; end
A = abs(M);
if strcmpi(method, 'area')
    A = resize_area(A, img_size);                       % 先面积平均再压缩
    A = log1p(A / (max(A(:)) + eps) * 100) / log1p(100);
else
    A = log1p(A / (max(A(:)) + eps) * 100) / log1p(100);
    A = resize_bilinear(A, img_size);
end
A = min(max(A, 0), 1);
idx = 1 + round(A * (size(cmap, 1) - 1));
img = reshape(cmap(idx, :), [img_size 3]);
end
