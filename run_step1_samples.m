%RUN_STEP1_SAMPLES  模块二：从原始数据构建带标签的样本集（替代 data_summary.m）
%   东南大学齿轮箱数据：data.mat 中 csv_acc_data{行,3} 为各状态的多通道数据。
%   接入自有矿区音频时，参考文末示例改为逐类读取音频文件。
clear; clc;
setup_paths();
cfg = config(); s = cfg.smp;

nC = numel(s.class_rows);
data = zeros(nC * s.n_per_class, s.win_len);
label = zeros(nC * s.n_per_class, 1);
raw = load(s.raw_file);
for c = 1:nC
    x = raw.csv_acc_data{s.class_rows(c), 3}(s.skip_head:end, s.channel);
    D = make_windows(x, s.win_len, s.win_step, s.n_per_class);
    rows = (c-1) * s.n_per_class + (1:s.n_per_class);
    data(rows, :) = D; label(rows) = c;
    fprintf('类别 %d (%s)：%d 个样本\n', c, s.class_names{c}, s.n_per_class);
end
Rawdata = [data label];            % 兼容旧格式：最后一列为标签
class_names = s.class_names;
save(s.out_file, 'Rawdata', 'class_names', '-v7');
fprintf('已保存 %s（%d x %d）\n', s.out_file, size(Rawdata, 1), size(Rawdata, 2));

% ---- 接入自有音频的写法示例 ----
% files = {'data/正常作业.wav', 'data/异常作业.wav'};
% for c = 1:numel(files)
%     [x, fs] = load_signal(files{c});
%     D = make_windows(x, s.win_len, s.win_step, s.n_per_class);
%     rows = (c-1) * s.n_per_class + (1:s.n_per_class);
%     data(rows, :) = D; label(rows) = c;
% end
