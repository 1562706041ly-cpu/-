function [x, fs] = load_signal(file, fs_default)
%LOAD_SIGNAL  读取一维信号，支持 .csv/.txt/.xlsx/.xls/.mat/.wav/.flac
%   .mat 文件取第一个数值变量；音频文件自动转单声道并返回真实采样率。
if nargin < 2, fs_default = 1; end
if ~exist(file, 'file'), error('找不到信号文件: %s', file); end
fs = fs_default;
[~, ~, ext] = fileparts(file);
switch lower(ext)
    case {'.csv', '.txt'}
        x = dlmread(file);
    case {'.xlsx', '.xls'}
        x = readmatrix(file);
    case '.mat'
        s = load(file); fn = fieldnames(s);
        k = find(cellfun(@(f) isnumeric(s.(f)) && numel(s.(f)) > 1, fn), 1);
        if isempty(k), error('MAT 文件中没有数值信号'); end
        x = s.(fn{k});
        if isfield(s, 'fs'), fs = s.fs; end
    case {'.wav', '.flac', '.mp3', '.ogg'}
        [x, fs] = audioread(file);
        x = mean(x, 2);                 % 多声道取平均
    otherwise
        error('不支持的文件类型: %s', ext);
end
x = double(x);
if ~isvector(x), x = x(:, 1); end
x = x(:);
x = x(isfinite(x));
end
