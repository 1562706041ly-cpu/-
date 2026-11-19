function D = make_windows(x, win_len, win_step, n_win)
%MAKE_WINDOWS  滑动窗口切分样本，返回 n_win x win_len 矩阵（预分配，替代循环拼接）
x = x(:);
need = (n_win - 1) * win_step + win_len;
if numel(x) < need
    error('信号长度 %d 不足，生成 %d 个样本至少需要 %d 个点', numel(x), n_win, need);
end
idx = (0:n_win-1).' * win_step + (1:win_len);
D = x(idx);
end
