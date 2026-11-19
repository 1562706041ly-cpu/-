function z = analytic_signal(x)
%ANALYTIC_SIGNAL  FFT 法求解析信号，等价于 hilbert(x)
x = x(:).';
n = numel(x);
h = zeros(1, n);
h(1) = 1;
if mod(n, 2) == 0
    h(n/2+1) = 1; h(2:n/2) = 2;
else
    h(2:(n+1)/2) = 2;
end
z = ifft(fft(x) .* h);
end
