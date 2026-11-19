function [u, omega, n_iter] = vmd_fast(signal, alpha, tau, K, DC, init, tol, max_iter)
%VMD_FAST  省内存版变分模态分解（与 Dragomiretskiy & Zosso 2014 算法一致）
%   原版 VMD.m 会保存全部 500 次迭代的频谱 (500 x 2N x K)，长音频会占用数 GB 内存；
%   本实现只保留当前迭代，计算结果与原版一致，内存占用约为原来的 1/500。
%   奇数长度信号会自动补齐一个点，输出再截回原长度。
if nargin < 8 || isempty(max_iter), max_iter = 500; end
x = signal(:).';
L = numel(x);
odd = mod(L, 2) == 1;
if odd, x = [x x(end)]; end
T0 = numel(x);
% 镜像延拓
f = [x(T0/2:-1:1), x, x(T0:-1:T0/2+1)];
T = numel(f);
t = (1:T) / T;
freqs = t - 0.5 - 1/T;
f_hat_plus = fftshift(fft(f));
f_hat_plus(1:T/2) = 0;
half = T/2+1:T;

u_hat = zeros(K, T);
switch init
    case 1, omega = (0.5/K) * (0:K-1);
    case 2, omega = sort(exp(log(1/T0) + (log(0.5)-log(1/T0)) * rand(1, K)));
    otherwise, omega = zeros(1, K);
end
if DC, omega(1) = 0; end
lambda_hat = zeros(1, T);
uDiff = tol + eps;
n_iter = 0;
while uDiff > tol && n_iter < max_iter - 1
    u_old = u_hat;
    sum_all = sum(u_hat, 1);
    for k = 1:K
        sum_uk = sum_all - u_hat(k, :);
        u_hat(k, :) = (f_hat_plus - sum_uk - lambda_hat/2) ./ (1 + alpha*(freqs - omega(k)).^2);
        sum_all = sum_uk + u_hat(k, :);
        if ~(DC && k == 1)
            p = abs(u_hat(k, half)).^2;
            omega(k) = (freqs(half) * p.') / sum(p);
        end
    end
    lambda_hat = lambda_hat + tau * (sum_all - f_hat_plus);
    n_iter = n_iter + 1;
    d = u_hat - u_old;
    uDiff = abs(eps + sum(sum(d .* conj(d))) / T);
end
% 重建时域信号
full_hat = zeros(T, K);
full_hat(half, :) = u_hat(:, half).';
full_hat(T/2+1:-1:2, :) = conj(u_hat(:, half).');
full_hat(1, :) = conj(full_hat(end, :));
u = real(ifft(ifftshift(full_hat, 1), [], 1)).';
u = u(:, T/4+1:3*T/4);
if odd, u = u(:, 1:L); end
% 按中心频率从低到高排序
[omega, idx] = sort(omega);
u = u(idx, :);
end
