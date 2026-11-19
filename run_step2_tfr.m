%RUN_STEP2_TFR  模块三：样本 -> 同步提取变换时频图（替代 Step1.m 与 Step2_JPG_Resize.m）
%   直接在内存中生成固定尺寸 RGB 图像，不再弹出 figure、保存 JPG 再读回；
%   同时修复原 Step1.m 中 axis_control(flag) 的 flag 未定义错误。
clear; close all; clc;
setup_paths();
cfg = config(); t = cfg.tfr;
S = load(cfg.smp.out_file);
data = S.Rawdata(:, 1:end-1); label = S.Rawdata(:, end);
n = size(data, 1);
resizeimg = cell(2, n);
if t.save_jpg, ensure_dir(t.jpg_dir); end
t0 = tic;
for i = 1:n
    Te = set_transform(data(i, :), t.voices, t.wav_param);
    img = tfr_to_rgb(Te, t.img_size, [], t.resize);
    resizeimg{1, i} = img;
    resizeimg{2, i} = label(i);
    if t.save_jpg, imwrite(img, fullfile(t.jpg_dir, sprintf('%04d_c%d.png', i, label(i)))); end
    if mod(i, 100) == 0, fprintf('%d / %d (%.0f s)\n', i, n, toc(t0)); end
end
save(t.out_file, 'resizeimg', '-v7');
fprintf('已保存 %s\n', t.out_file);
