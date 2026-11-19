function setup_paths(varargin)
%   setup_paths('lssvm') 额外加入 LSSVMlab（其 range.m 会遮蔽同名函数，仅分类模块需要）
%SETUP_PATHS  加入自研代码与第三方依赖路径
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));
addpath(fullfile(root, 'third_party', 'TTAO'));
addpath(fullfile(root, 'third_party', 'GLO'));
addpath(fullfile(root, 'third_party', 'VMD'));
if any(strcmpi(varargin, 'lssvm'))
    addpath(fullfile(root, 'third_party', 'LSSVMlab'));
end
end
