function L = compute_normalized_laplacian(H, weights, k)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 计算归一化拉普拉斯矩阵
    % 输入：
    %   H      : 关联矩阵 (N×M)
    %   weights: 超边权重 (M×1)
    %   k      : 超边大小
    % 输出：归一化拉普拉斯矩阵 (N×N)
    
    W = spdiags(weights, 0, length(weights), length(weights)); % 超边权重矩阵 (M×M)
    Dv = sum(H, 2); % 顶点度矩阵 (N×1)
    Dv = spdiags(1 ./ (sqrt(Dv)+1e-2), 0, size(Dv,1), size(Dv,1)); % 顶点度矩阵的逆平方根 Dv^{-1/2} (N×N)
    De = 1/k * speye(size(H,2)); % 超边度矩阵 (k均匀超图) (M×M) De^-1
    
    % 计算归一化拉普拉斯矩阵: L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    L = speye(size(Dv)) - Dv * H * W * De * (H') * Dv; % (N×N)
    L = single(full(L));
end