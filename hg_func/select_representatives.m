function representatives = select_representatives(clusters)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 从每个类中均匀选取3个代表点
    % 输入：clusters 结构数组
    % 输出：代表点全局索引 (num_clusters×3)
    
    num_clusters = length(clusters);
    representatives = zeros(num_clusters, 3);
    
    for i = 1:num_clusters
        n = clusters{i}.size;
        indices = clusters{i}.indices;
        
        step = max(1, floor((n-1)/2));
        reps = indices([1, 1+step, min(n, 1+2*step)]);
        representatives(i, :) = reps;
    end
end