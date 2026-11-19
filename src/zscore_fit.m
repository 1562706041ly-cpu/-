function [Xs, mu, sd] = zscore_fit(X)
%ZSCORE_FIT  用训练集统计量标准化（测试集请用同一 mu/sd）
mu = mean(X, 1); sd = std(X, 0, 1); sd(sd == 0) = 1;
Xs = (X - mu) ./ sd;
end
