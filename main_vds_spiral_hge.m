%% Main function for VDS Spiral HGE
% Author: Peng Li
% Date: 2025/06/08
% Description: This script is used to RUN the HGE algorithm under VDS Spiral  Undersampling Pattern.
clc;
close all;
clear all;
clear classes;
%% add path
addpath('py_func/')
addpath('utils/');
addpath('hg_func/')
%% 初始化参数设置
% 数据参数
para.res = [128, 128, 500]; % 图像大小 [Nx, Ny, L]
save_root = ['results/', num2str(para.res(3)), '_vds_noisy/'];
if ~exist(save_root, 'dir')
    mkdir(save_root);
end

para.noisy = 1; %是否添加噪声
para.noise_level = 0.1; % 噪声水平 SNR 23dB
para.PD_norm = 116.8877;
% dictionary matching参数
para.matching_var = 0.9; % 匹配阈值
para.matching_batch = 10000; % 同时最大匹配条目batch size
% 超图参数
para.k = 30; % 超边大小
para.cluster_num = 10; % 聚类的数量
para.showFlag = 0; % 是否显示结果
% 迭代算法参数
para.maxIter = 50; % 最大迭代次数
para.threshold = 1e-3; % iteration stopping criterion
para.acc = 1; % acceleration flag
para.t = 1; % parameter for the acceleration
para.lambda = 1e-3; % regularization parameter
para.mu_min = 0.5; % iteration step minimum and initial value
para.mu_max = 2.0; % maximum iteration step
para.print_snr = 1; % calulate and print snr
%%  load py functions
py.importlib.reload(py.importlib.import_module('py_func.dic_matching'));
py.importlib.reload(py.importlib.import_module('py_func.build_dic_matrix'));
py.importlib.reload(py.importlib.import_module('py_func.nufft_for_matlab')); %single coil
%% 数据加载
load('data/input_to_fisp_experiment.mat') %仿真参数图 T1_128 T2_128 PD_128
load("data/fatr.mat") %FISP序列参数 fa tr
load('data/vds_spiral_ktraj.mat') %variable density spiral变密度螺旋采样模板 ktraj
fa = fa(1:para.res(3));
tr = tr(1:para.res(3));
ktraj = ktraj(1:para.res(3), :, :) * 2 * pi; %to pi
%% nufft initialization初始化NUFFT
torch_batch_size = 1000; %nufft使用PyTorch实现，batch size越大速度越快，需显存越大
grid_factor = 2; % nufft grid factor  越大精度越高，计算开销越大
py.py_func.nufft_for_matlab.init_nufft_op(torch_batch_size, para.res(1), para.res(2), para.res(3), grid_factor, ...
py.numpy.array(ktraj)) %single coil
%% build Dictionary
tic;
tmp = py.py_func.build_dic_matrix.build_dictionary_mat(py.numpy.array(fa), py.numpy.array(tr));
D = double(tmp{'dic'});
D = single(D);
LUT = double(tmp{'lut'});
time_build_D = toc;
fprintf('build Dictionary time: %.4f s; \n', time_build_D)
%% build X
tic;
tmp = py.py_func.build_dic_matrix.build_TemplateMatrix_mat(py.numpy.array(fa), ...
    py.numpy.array(tr), py.numpy.array(T1_128), py.numpy.array(T2_128), py.numpy.array(PD_128));
X = double(tmp);
X = single(X);
para_maps = cat(3, T1_128, T2_128, PD_128);
para_maps_mask = cat(3, T1_128 > 10, T2_128 > 10, PD_128 > 0);
time_build_X = toc;
fprintf('build X time: %.4f s; \n', time_build_X)
% 归一化
para.m_abs = max(abs(X(:))); % 归一化
Y = fft2(X ./ para.m_abs);
%% 定义字典匹配算子、投影算子、欠采样算子
[dic_matching, Proj_D] = def_matching(D, LUT, para.matching_var, para.matching_batch);
para_maps_gt = dic_matching(X);
para_maps_gt = para_maps_gt .* para_maps_mask;
[A, At, AtA] = defAAt_spiral(para.m_abs);
[P, Pt] = def_PPt(para.res); % PPt operator for X
%% 增加噪声
Y_down = A(X);
if para.noisy == 1
    kSpaceNoise = reshape([1, 1i] * para.noise_level * randn(2, para.res(1) * para.res(2) * para.res(3)), para.res);
    n = A(kSpaceNoise);
    % noise SNR
    SNR_noisy = 20 * log10(norm(Y_down(:)) / norm(n(:)));
    fprintf('SNR_noisy: %.4f dB; \n', SNR_noisy)
    Y_down = Y_down + n;
end
%% HGE 算法
[X_recon_hge, para_maps_recon_hge, SNR_hge, Loss_hge, time_hge] = hge_solver(Y_down, X, Proj_D, A, At, AtA,P, Pt, dic_matching, para);
para_maps_recon_hge = para_maps_recon_hge.* para_maps_mask;
%% 显示结果
map_range.T1_value = 2500;
map_range.T1_value_error = 500;
map_range.T2_value = 500;
map_range.T2_value_error = 200;
map_range.PD_value = 116.8877;
map_range.PD_value_error = 23.3775;
num_maps = 3;
imagesc_para(map_range, num_maps, para_maps_gt, {'HGE'}, para_maps_recon_hge)
%% 保存结果
save_results_img(map_range, num_maps, save_root, para_maps_gt, {'HGE'}, para_maps_recon_hge);
save(save_root + "results.mat", 'X_recon_hge', 'para_maps_recon_hge', 'SNR_hge', 'Loss_hge', ...
    'time_hge', 'SNR_noisy', 'para_maps_gt');
