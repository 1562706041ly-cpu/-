classdef ecaLayer < nnet.layer.Layer
%ECALAYER  ECA 通道注意力层（Wang et al., CVPR 2020）自研实现
%   全局平均池化 -> 跨通道 1D 卷积（核长 3，无偏置）-> sigmoid -> 按通道加权。
%   通道数 16/32/64 时论文公式给出的核长都是 3，每层只有 3 个可学习参数。
%   与 bench/models.py 中的 ECA 类一一对应。需要 Deep Learning Toolbox（R2021a 及以上）。
    properties (Learnable)
        Weights     % 3 x 1，依次作用于第 c-1、c、c+1 个通道
    end
    methods
        function layer = ecaLayer(name)
            layer.Name = name;
            layer.Description = 'ECA channel attention (k=3)';
            layer.Weights = (rand(3, 1) * 2 - 1) / sqrt(3);
        end
        function Z = predict(layer, X)
            % X: H x W x C x N
            C = size(X, 3);
            m = mean(X, [1 2]);                    % 1 x 1 x C x N
            prev = 0 * m; next = 0 * m;            % 边界补 0
            if C > 1
                prev(:, :, 2:C, :) = m(:, :, 1:C-1, :);
                next(:, :, 1:C-1, :) = m(:, :, 2:C, :);
            end
            w = layer.Weights;
            a = w(1) * prev + w(2) * m + w(3) * next;
            Z = X .* (1 ./ (1 + exp(-a)));
        end
    end
end
