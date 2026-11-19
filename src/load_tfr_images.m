function [X, y] = load_tfr_images(file)
%LOAD_TFR_IMAGES  读取时频图库，返回 X (64x64x3xN double) 和 y (N x 1，1..C)
%   支持两种格式：
%   1) run_step2_tfr 生成的 resizeimg.mat（2 x N 元胞：RGB 图像 + 标签）；
%   2) bench/make_banks.py 生成的图库 .mat（X: N x 64 x 64 uint8 颜色索引 0..255，y: N x 1）。
%      颜色索引经 jet(256) 映射后与 tfr_to_rgb 的输出逐像素相同。
S = load(file);
if isfield(S, 'resizeimg')
    imgs = S.resizeimg(1, :);
    y = double(cell2mat(S.resizeimg(2, :))).';
    X = cat(4, imgs{:});
else
    J = jet(256);
    idx = double(S.X) + 1;                 % N x H x W
    [N, H, W] = size(idx);
    rgb = reshape(J(idx(:), :), [N H W 3]);
    X = permute(rgb, [2 3 4 1]);           % H x W x 3 x N
    y = double(S.y(:));
end
end
