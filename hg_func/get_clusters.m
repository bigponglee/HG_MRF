function [clusters, unique_labels] = get_clusters(X, labels)
    % Author: Peng Li
    % Date: 2025/06/08  
    % 函数功能:
    % 获取聚类以及调整后的详细信息
    % 输入：
    %   X     : 数据矩阵
    %   labels: 调整后的聚类标签
    % 输出：
    %   clusters: 包含每个类数据的元胞数组
    %   unique_labels: 唯一的类标签
    
    unique_labels = unique(labels);
    clusters = cell(length(unique_labels), 1);
    
    for i = 1:length(unique_labels)
        idx = (labels == unique_labels(i));
        clusters{i}.data = X(idx, :);
        clusters{i}.indices = find(idx); % 保存全局索引
        clusters{i}.size = sum(idx);
    end
end