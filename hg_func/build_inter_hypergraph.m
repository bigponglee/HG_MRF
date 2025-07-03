function [hyperedges, weights] = build_inter_hypergraph(X, representatives, k)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 构建类间超边（基于代表点）
    % 输入：
    %   X             : 所有数据 (N×d)
    %   representatives: 代表点索引 (C×3)
    %   k             : 超边大小
    % 输出：
    %   hyperedges: 超边列表 (C×k)，每行表示一条超边，元素为全局索引
    %   weights   : 超边权重 (C×1)
    
    all_reps = representatives(:); % 所有代表点 (3C×1)
    num_reps = length(all_reps);
    num_hyperedges = num_reps; % 每个代表点都可以形成一个超边
    
    hyperedges = zeros(num_hyperedges, k);
    weights = zeros(num_hyperedges, 1);
    
    rep_data = X(all_reps, :);
    dists_inter = pdist2(rep_data, rep_data); % 计算类间代表点之间的距离
    sigma = median(dists_inter(:)); % 使用中位数作为距离尺度
    if sigma == 0
        sigma = 1; % 防止除以零
    end
    for i = 1:num_hyperedges
        % 基于代表点距离选取k个最近邻点
        dists = dists_inter(i, :); % 当前代表点到所有其他代表点的距离
        [~, idxs] = mink(dists, k + 1); % 包含自身的k近邻
        idxs = idxs(1:k); % 排除自身

        hyperedges(i, :) = all_reps(idxs); % 全局索引
        % 计算超边权重：基于平均距离
        avg_dist = mean(dists(idxs(2:end))); % 排除自身
        weights(i) = exp(-avg_dist/sigma); % 指数衰减
    end
end