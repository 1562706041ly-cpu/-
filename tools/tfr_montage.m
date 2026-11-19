function tfr_montage(files, out_png, idx, scale)
%TFR_MONTAGE  把几组时频图拼成一张对比图（每行一组，每列一个样本），放大 scale 倍保存
%   例：tfr_montage({'data/resizeimg_old.mat','data/resizeimg.mat'}, 'results/tfr_compare.png', [1 201 401 601 801], 4)
if nargin < 4, scale = 4; end
rows = {};
for f = 1:numel(files)
    S = load(files{f});
    tiles = cellfun(@(i) pad(S.resizeimg{1, i}), num2cell(idx), 'UniformOutput', false);
    rows{end+1} = cat(2, tiles{:}); %#ok<AGROW>
end
M = cat(1, rows{:});
M = M(kron(1:size(M, 1), ones(1, scale)), kron(1:size(M, 2), ones(1, scale)), :);
imwrite(M, out_png);
end

function T = pad(img)
T = ones(size(img, 1) + 4, size(img, 2) + 4, 3);
T(3:end-2, 3:end-2, :) = img;
end
