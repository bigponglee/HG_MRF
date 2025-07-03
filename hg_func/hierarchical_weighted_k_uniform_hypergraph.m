function L_normalized = hierarchical_weighted_k_uniform_hypergraph(X, P, para)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 主函数：构建层级加权k均匀超图并计算归一化拉普拉斯矩阵
    % 输入：
    %   X : 输入参数图  N*N*3
    %   P: 前向算子函数
    %   para : 参数结构体，包含以下字段：
    %       k : 超边大小 (标量)
    %       cluster_num : 初始聚类数 (标量)
    %       show_flag : 是否可视化 (布尔值)
    % 输出：
    %   L_normalized : 归一化拉普拉斯矩阵 (N×N)

    % 参数解析
    k = para.k; % 超边大小
    cluster_num = para.cluster_num; % 初始聚类数

    X = P(X(:, :, 1:2)); % 前向算子处理输入参数图，这里只使用前两个通道 T1 和 T2

    % 1. 初始聚类划分
    cluster_labels = initial_clustering(X, cluster_num);
    
    % 2. 调整类大小（分裂过大类，合并过小类）
    adjusted_labels = adjust_cluster_sizes(X, cluster_labels, k);
    
    % 3. 获取聚类信息
    [clusters, unique_labels] = get_clusters(X, adjusted_labels);
    num_clusters = length(unique_labels);
    
    % 4. 构建类内加权子超图
    intra_hyperedges = cell(num_clusters, 1);
    intra_weights = cell(num_clusters, 1);
    for i = 1:num_clusters
        cluster_data = clusters{i}.data;
        [intra_hyperedges{i}, intra_weights{i}] = build_intra_hypergraph(cluster_data, k);
        % 将类内索引转换为全局索引
        intra_hyperedges{i} = clusters{i}.indices(intra_hyperedges{i});
    end
    
    % 5. 构建类间超边（使用代表点）
    representatives = select_representatives(clusters); % 每个类选3个代表点
    [inter_hyperedges, inter_weights] = build_inter_hypergraph(X, representatives, k);
    
    % 6. 合并所有超边
    all_hyperedges = [vertcat(intra_hyperedges{:}); inter_hyperedges];
    all_weights = [vertcat(intra_weights{:}); inter_weights];
    
    % 7. 构建超图关联矩阵
    H = build_incidence_matrix(size(X,1), all_hyperedges, k);
    
    % 8. 计算归一化拉普拉斯矩阵
    L_normalized = compute_normalized_laplacian(H, all_weights, k);

    if para.showFlag
        figure;
        subplot(2, 2, 1);
        scatter(X(:,1), X(:,2), 10, cluster_labels, 'filled');
        title('Initial Clustering');
        xlabel('T1 值');
        ylabel('T2 值');
        title('K-means 聚类结果');
        % 将聚类分配结果重塑为与原始图像相同的二维形状
        cluster_image = reshape(cluster_labels, para.res(1), para.res(2));
        subplot(2, 2, 2);
        imagesc(cluster_image);
        colormap(turbo); 
        colorbar; % 显示颜色条
        title('K-means 聚类结果');
        axis image;
        subplot(2, 2, 3);
        scatter(X(:,1), X(:,2), 10, adjusted_labels, 'filled');
        title('Adjusted Clustering');
        xlabel('T1 值');
        ylabel('T2 值');
        title('K-means 聚类结果')
        % 将聚类分配结果重塑为与原始图像相同的二维形状
        cluster_image = reshape(adjusted_labels, para.res(1), para.res(2));
        subplot(2, 2, 4);
        imagesc(cluster_image);
        colormap(turbo); 
        colorbar; % 显示颜色条
        title('K-means 聚类结果');
        axis image;
        % % 保存cluster_image图像，转化为彩色图像’
        % cluster_image = ind2rgb(cluster_image, turbo(max(unique_labels(:)))); % 将索引图像转换为RGB图像
        % save_imshow(cluster_image,['results//500_vds_noisy//cluster//colored_cluster_', num2str(save_index)]);
    end
end