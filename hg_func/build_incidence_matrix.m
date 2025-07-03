function H = build_incidence_matrix(N, hyperedges, k)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    % 构建超图关联矩阵
    % 输入：
    %   N         : 顶点数
    %   hyperedges: 超边列表 (M×k)
    %   weights   : 超边权重 (M×1)
    %   k         : 超边大小
    % 输出：
    %   H         : 稀疏关联矩阵 (N×M)
    %   W         : 超边权重 (M×1)
    
    M = size(hyperedges, 1);
    rows = [];
    cols = [];
    vals = [];
    
    for e = 1:M
        vertices = hyperedges(e, :);
        
        rows = [rows; vertices(:)];
        cols = [cols; e * ones(k, 1)];
        vals = [vals; ones(k, 1)];
    end
    
    % 创建稀疏矩阵
    H = sparse(rows, cols, vals, N, M);
end