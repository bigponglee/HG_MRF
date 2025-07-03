function PtPxL = manifold_cg(x, L_normalized, P, P_t, res)
    % Author: Peng Li
    % Date: 2025/06/08
    % 函数功能:
    %   求解流形约束项
    % 输入参数:
    %   x: 输入图像，[Nx, Ny, L]
    %   L_normalized: 归一化拉普拉斯矩阵 NxNy x NxNy
    %   P: 前向算子函数
    %   P_t: 后向算子函数
    %   res: 图像大小 [Nx, Ny, L]
    % 输出参数:
    %   PtPxL: 流形约束项，[Nx, Ny, L]

    if ~isequal(size(x), res)
        x = reshape(x, res);
    end
    
    Px = P(x); % NxNy x L

    PxL = L_normalized * Px; % NxNy x L

    PtPxL = P_t(PxL); % [N,N,L]

end
