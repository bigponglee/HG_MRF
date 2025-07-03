function adjusted_labels = adjust_cluster_sizes(X, labels, k)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 调整类大小：分裂过大类(>100k)，合并过小类(<k)
    % 输入：
    %   X     : 数据矩阵 (N×d)
    %   labels: 初始聚类标签
    %   k     : 超边大小
    % 输出：调整后的聚类标签
    
    adjusted_labels = labels;
    max_size = 100 * k; % 类大小上限
    
    % 内部函数：计算聚类信息
    function [unique_labels, cluster_info] = get_cluster_info(labels_t)
        unique_labels = unique(labels_t);
        n = length(unique_labels);
        cluster_info = struct('size', cell(1,n), 'indices', cell(1,n), 'center', cell(1,n));
        for i_t = 1:n
            idx = (labels_t == unique_labels(i_t));
            cluster_info(i_t).size = sum(idx);
            cluster_info(i_t).indices = find(idx);
            cluster_info(i_t).center = mean(X(idx, :), 1);
        end
    end
    
    % 分裂过大的类 (>100k)
    [unique_labels, cluster_info] = get_cluster_info(adjusted_labels);
    huge_clusters = find([cluster_info.size] > max_size);
    next_label = max(adjusted_labels) + 1;
    for i = 1:length(huge_clusters)
        idx_huge = huge_clusters(i);
        indices = cluster_info(idx_huge).indices;
        num_splits = ceil(cluster_info(idx_huge).size / max_size);
        sub_labels = ceil((1:cluster_info(idx_huge).size)' / max_size);
        for j = 1:num_splits
            if j == 1
                adjusted_labels(indices(sub_labels == j)) = unique_labels(idx_huge);
            else
                adjusted_labels(indices(sub_labels == j)) = next_label;
                next_label = next_label + 1;
            end
        end
    end

    % 2. 合并过小的类
    while true
        [unique_labels, cluster_info] = get_cluster_info(adjusted_labels);
        sizes = [cluster_info.size];
        small_clusters = find(sizes < k + 1);
        if isempty(small_clusters)
            break;
        end
        centers = vertcat(cluster_info.center);
        idx_small = small_clusters(1);
        dists = pdist2(centers, centers(idx_small, :));
        dists(idx_small) = inf;
        [~, nearest_idx] = min(dists);
        adjusted_labels(adjusted_labels == unique_labels(idx_small)) = unique_labels(nearest_idx);
    end
end