function cfg = config()
%CONFIG  矿盾声鉴算法模块统一配置（所有可调参数集中在这里）
%   cfg = config() 返回结构体，各运行脚本只从这里读取参数。

root = fileparts(mfilename('fullpath'));
cfg.root        = root;
cfg.data_dir    = fullfile(root, 'data');
cfg.result_dir  = fullfile(root, 'results');
cfg.seed        = 42;            % 全局随机种子，保证结果可复现
cfg.save_figures = true;

%% ---------------- 模块一：VMD 参数寻优 ----------------
cfg.vmd.signal_file = fullfile(cfg.data_dir, 'signal_demo.csv'); % 支持 .csv/.txt/.xlsx/.mat/.wav
cfg.vmd.fs          = 1;          % 采样率(Hz)。填实际值，频谱横轴才是真实频率
cfg.vmd.alpha_range = [100 2500]; % 惩罚因子 alpha 搜索范围
cfg.vmd.K_range     = [3 10];     % 模态数 K 搜索范围（整数）
cfg.vmd.tau  = 0;
cfg.vmd.DC   = 0;
cfg.vmd.init = 1;
cfg.vmd.tol  = 1e-7;
cfg.vmd.max_inner_iter = 500;
cfg.vmd.fitness   = 'envelope';   % 'envelope' | 'sample' | 'permutation' | 'info'
cfg.vmd.aggregate = 'min';        % 各 IMF 指标的汇总方式：'min'(局部极小) | 'mean'
cfg.vmd.optimizer = 'TTAO';       % 'TTAO' | 'GLO' | 'RANDOM'
cfg.vmd.pop_size  = 12;           % 种群规模（TTAO 建议为 3 的倍数）
cfg.vmd.max_iter  = 20;           % 迭代次数
cfg.vmd.ttao_step = 0.15;         % TTAO 三角单元尺寸（相对搜索范围的比例）

%% ---------------- 算法对比实验 ----------------
cfg.cmp.optimizers = {'TTAO', 'GLO', 'RANDOM'};
cfg.cmp.n_runs     = 20;          % 每个算法独立运行次数（论文/答辩建议 >= 20）
cfg.cmp.max_evals  = 200;         % 统一的适应度评估预算，保证公平
cfg.cmp.do_grid    = true;        % 是否用网格穷举得到参考最优值

%% ---------------- 模块二：样本构建（Step1） ----------------
cfg.smp.raw_file    = fullfile(cfg.data_dir, 'data.mat'); % 东南大学原始数据或自有数据
cfg.smp.channel     = 3;          % 选用的通道
cfg.smp.skip_head   = 1000;       % 去掉开头的点（去除端点效应）
cfg.smp.win_len     = 2048;       % 样本长度
cfg.smp.win_step    = 1000;       % 滑窗步长
cfg.smp.n_per_class = 200;        % 每类样本数
cfg.smp.class_rows  = [1 3 5 7 9];% csv_acc_data 中各类别所在行
cfg.smp.class_names = {'Chipped','Health','Miss','Root','Surface'};
cfg.smp.out_file    = fullfile(cfg.data_dir, 'data_total.mat');

%% ---------------- 模块三：时频图（Step2） ----------------
cfg.tfr.img_size  = [64 64];
cfg.tfr.voices    = 32;           % 每倍频程音数
cfg.tfr.wav_param = 6;            % Morlet 中心频率
cfg.tfr.resize    = 'area';       % 缩放方式：'area'(面积平均，推荐) | 'bilinear'(v2.0 逐点取样，会丢信息)
cfg.tfr.out_file  = fullfile(cfg.data_dir, 'resizeimg.mat');
cfg.tfr.save_jpg  = false;        % 需要展示图片时设为 true
cfg.tfr.jpg_dir   = fullfile(cfg.data_dir, 'tfr_images');

%% ---------------- 模块四：CNN-BKA-LSSVM（Step3） ----------------
cfg.cls.image_file  = cfg.tfr.out_file;
cfg.cls.split       = [0.6 0.2 0.2];  % 每类按时间顺序划分 训练/验证/测试
cfg.cls.gap         = 2;              % 划分边界丢弃的样本数，避免重叠窗口泄露
cfg.cls.max_epochs  = 60;
cfg.cls.batch       = 32;
cfg.cls.lr          = 1e-3;
cfg.cls.patience    = 8;              % 验证集早停耐心值
cfg.cls.exec_env    = 'auto';         % 'auto' | 'cpu' | 'gpu'
cfg.cls.feat_layer  = 'fc1';          % 送入 LSSVM 的特征层（64 维）
cfg.cls.bka_pop     = 12;
cfg.cls.bka_iter    = 20;
cfg.cls.log10_gam   = [-2 6];         % 在对数空间搜索 gam（实测最优常在 1e3~1e4，上限留足）
cfg.cls.log10_sig2  = [-2 4];         % 在对数空间搜索 sig2
cfg.cls.lssvm_impl  = 'fast';         % 'fast'(自研向量化实现) | 'lssvmlab'(第三方库)
cfg.cls.cv_folds    = 5;              % 只在训练+验证集内做 k 折交叉验证
cfg.cls.n_repeats   = 5;              % 重复实验次数，报告均值±标准差
end
