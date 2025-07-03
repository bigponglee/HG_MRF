function [hyperedges, weights] = build_intra_hypergraph(cluster_data, k)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 构建类内加权k均匀超图
    % 输入：
    %   cluster_data: 类内数据 (n×d)
    %   k          : 超边大小
    % 输出：
    %   hyperedges: 超边列表 (n×k)，每行表示一条超边，元素为类内相对索引
    %   weights   : 超边权重 (n×1)
    
    n = size(cluster_data, 1);
    num_hyperedges = n; % 每个点都可以形成一个超边
    
    hyperedges = zeros(num_hyperedges, k);
    weights = zeros(num_hyperedges, 1);
    dists_intra = pdist2(cluster_data, cluster_data); % 计算类内点之间的距离
    sigma = median(dists_intra(:)); % 使用中位数作为距离尺度
    if sigma == 0
        sigma = 1; % 防止除以零
    end
    
    for i = 1:num_hyperedges
        % 基于类内距离选取k个最近邻点
        dists = dists_intra(i, :); % 当前点到所有其他点的距离
        [~, idxs] = mink(dists, k + 1); % 包含自身的k近邻
        idxs = idxs(1:k); % 排除自身

        hyperedges(i, :) = idxs; % 类内相对索引，需要转换为实际数据索引
    
        % 计算超边权重：基于平均距离
        avg_dist = mean(dists(idxs(2:end))); % 排除自身
        weights(i) = exp(-avg_dist/sigma); % 指数衰减
    end
end