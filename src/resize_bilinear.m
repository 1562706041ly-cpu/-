function B = resize_bilinear(A, sz)
%RESIZE_BILINEAR  双线性缩放（不依赖图像处理工具箱）
[h, w] = size(A);
[xq, yq] = meshgrid(linspace(1, w, sz(2)), linspace(1, h, sz(1)));
B = interp2(A, xq, yq, 'linear');
end
