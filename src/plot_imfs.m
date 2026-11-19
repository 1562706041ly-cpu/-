function fig = plot_imfs(u, x, fs, ttl)
%PLOT_IMFS  绘制原信号、各 IMF 时域波形及其幅值谱（VMD 无残差项）
K = size(u, 1);
t = (0:size(u, 2)-1) / fs;
fig = figure('Color', 'w', 'Name', ttl, 'Position', [100 80 900 120*(K+1)]);
subplot(K+1, 2, 1); plot(t, x, 'k'); axis tight; ylabel('原信号'); title([ttl ' 时域']);
[a, f] = amp_spectrum(x, fs);
subplot(K+1, 2, 2); plot(f, a, 'k'); axis tight; title('幅值谱');
for k = 1:K
    subplot(K+1, 2, 2*k+1); plot(t, u(k, :), 'b'); axis tight; ylabel(sprintf('IMF%d', k));
    [a, f] = amp_spectrum(u(k, :), fs);
    subplot(K+1, 2, 2*k+2); plot(f, a, 'r'); axis tight;
end
if fs == 1, xlabel('归一化频率'); else, xlabel('频率 / Hz'); end
end
