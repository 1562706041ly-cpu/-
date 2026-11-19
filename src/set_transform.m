function [Te, freq] = set_transform(x, nv, cf)
%SET_TRANSFORM  小波同步提取变换（Synchroextracting Transform, Yu Gang 2017）自研实现
%   只保留瞬时频率估计与尺度中心频率一致处的 CWT 系数，得到高能量集中度的时频图。
%   x  : 信号；nv : 每倍频程音数（默认 32）；cf : 解析 Morlet 中心频率（默认 6）
%   Te : na x N 时频矩阵（复数）；freq : 各行对应的归一化频率（周期/采样点）
if nargin < 2 || isempty(nv), nv = 32; end
if nargin < 3 || isempty(cf), cf = 6; end
x = x(:).' - mean(x);
N = numel(x);
noct = floor(log2(N)) - 1;
na = noct * nv;
scales = 2.^((1:na) / nv);
omega = (1:fix(N/2)) * (2*pi/N);
omega = [0, omega, -omega(fix((N-1)/2):-1:1)];
X = fft(x);
psi  = exp(-(scales.' * omega - cf).^2 / 2) .* (omega > 0);  % 解析 Morlet 频域
dpsi = 1i * omega .* psi;
W  = ifft(X .* psi, [], 2);
dW = ifft(X .* dpsi, [], 2);
% 瞬时频率估计（周期/采样点）
ifreq = imag(dW ./ (W + eps)) / (2*pi);
% 每个尺度的中心频率与频率分辨率
fc = cf ./ (2*pi*scales.');
df = fc * (2^(1/nv) - 1);
mask = abs(ifreq - fc) <= df / 2 & abs(W) > 1e-8 * max(abs(W(:)));
Te = W .* mask;
% 行从高频到低频，翻转成低频在下
Te = flipud(Te);
freq = flipud(fc);
end
