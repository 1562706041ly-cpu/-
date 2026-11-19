function fig = plot_confusion(C, names, ttl)
%PLOT_CONFUSION  混淆矩阵热力图（行：真实类别，列：预测类别）
n = size(C, 1);
R = C ./ max(sum(C, 2), 1);
fig = figure('Color', 'w', 'Position', [100 100 520 440]);
imagesc(R, [0 1]);
colormap([linspace(1, 0.08, 256).' linspace(1, 0.32, 256).' linspace(1, 0.65, 256).']);  % 白->蓝
colorbar; axis square;
for i = 1:n
    for j = 1:n
        if R(i, j) > 0.5, col = 'w'; else, col = 'k'; end
        text(j, i, sprintf('%d\n%.1f%%', C(i, j), 100 * R(i, j)), ...
            'HorizontalAlignment', 'center', 'Color', col, 'FontSize', 9);
    end
end
set(gca, 'XTick', 1:n, 'YTick', 1:n, 'XTickLabel', names, 'YTickLabel', names);
xlabel('预测类别'); ylabel('真实类别'); title(ttl);
end
