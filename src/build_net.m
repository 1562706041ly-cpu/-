function lg = build_net(name, input_size, n_classes)
%BUILD_NET  构建识别网络（返回 layerGraph，可直接传给 trainNetwork）
%   name：'cnn'        v2 基线 CNN
%         'resnet'     小型残差网络（3 个残差块，通道 16/32/64），用于消融
%         'eca_resnet' resnet + ECA 通道注意力（本次升级的主模型，需要 src/ecaLayer.m）
%   三种网络的分辨率路径相同（64 -> 32 -> 16 -> 全局平均池化），都有 64 维的 'fc1' 层，
%   因此后面的 BKA-LSSVM 阶段不用改。与 bench/models.py 中的结构一一对应。
head = [
    globalAveragePooling2dLayer('Name', 'gap')
    dropoutLayer(0.3, 'Name', 'drop')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    fullyConnectedLayer(n_classes, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
inp = imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'zscore');

switch lower(name)
    case 'cnn'
        lg = layerGraph([
            inp
            convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
            convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
            convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')
            head]);
    case {'resnet', 'eca_resnet'}
        eca = strcmpi(name, 'eca_resnet');
        lg = layerGraph([
            inp
            convolution2dLayer(3, 16, 'Padding', 'same', 'BiasLearnRateFactor', 0, 'Name', 'stem_conv')
            batchNormalizationLayer('Name', 'stem_bn')
            reluLayer('Name', 'stem_relu')]);
        last = 'stem_relu';
        cfgs = [16 16 1; 16 32 2; 32 64 2];          % 输入通道、输出通道、步长
        for b = 1:3
            [lg, last] = add_block(lg, last, sprintf('b%d', b), cfgs(b, 1), cfgs(b, 2), cfgs(b, 3), eca);
        end
        lg = addLayers(lg, head);
        lg = connectLayers(lg, last, 'gap');
    otherwise
        error('未知网络: %s', name);
end
end

function [lg, out] = add_block(lg, in, p, cin, cout, stride, eca)
% 残差块：conv3x3(stride)-BN-ReLU-conv3x3-BN-[ECA] + 捷径，再 ReLU
body = [
    convolution2dLayer(3, cout, 'Stride', stride, 'Padding', 'same', 'BiasLearnRateFactor', 0, 'Name', [p '_conv1'])
    batchNormalizationLayer('Name', [p '_bn1'])
    reluLayer('Name', [p '_relu1'])
    convolution2dLayer(3, cout, 'Padding', 'same', 'BiasLearnRateFactor', 0, 'Name', [p '_conv2'])
    batchNormalizationLayer('Name', [p '_bn2'])];
if eca
    body = [body; ecaLayer([p '_eca'])];
end
body = [body
    additionLayer(2, 'Name', [p '_add'])
    reluLayer('Name', [p '_relu'])];
lg = addLayers(lg, body);
lg = connectLayers(lg, in, [p '_conv1']);
if stride == 1 && cin == cout
    lg = connectLayers(lg, in, [p '_add/in2']);
else
    sc = [
        convolution2dLayer(1, cout, 'Stride', stride, 'BiasLearnRateFactor', 0, 'Name', [p '_sc_conv'])
        batchNormalizationLayer('Name', [p '_sc_bn'])];
    lg = addLayers(lg, sc);
    lg = connectLayers(lg, in, [p '_sc_conv']);
    lg = connectLayers(lg, [p '_sc_bn'], [p '_add/in2']);
end
out = [p '_relu'];
end
