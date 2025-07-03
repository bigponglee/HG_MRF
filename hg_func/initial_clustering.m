function labels = initial_clustering(X, k_opt)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 使用K-means进行初始聚类
    % 输入：数据矩阵 X (N×d)
    %        聚类数 k_opt (标量)
    % 输出：聚类标签 (N×1)

    % 执行K-means聚类
    [labels, ~] = kmeans(X, k_opt, 'Replicates', 5, 'MaxIter', 200);
end