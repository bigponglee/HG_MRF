function [P, Pt] = def_PPt(im_size)
    % def forward/backward patch operator
    % input: x - input image (3D array) size Nx x Ny x L
    % output: P - forward operator
    %         Pt - backward operator
    % Author: Peng Li
    % Date: 2025/06/08

    Nx = im_size(1);
    Ny = im_size(2);

    function patches = patch_image(x)
        % input x: image [Nx, Ny, L]
        % output patches: Forward operator output NxNy x L
        patches = reshape(x, [], size(x, 3)); % Reshape to NxNy x L        
    end

    function x = unpatch_image(patches)
        % input patches: Forward operator output NxNy x L
        % output x: reconstructed image [Nx, Ny, L]
        x = reshape(patches, [Nx, Ny, size(patches, 2)]); % Reshape back to Nx x Ny x L
    end

    % Define the forward operator
    P = @(x) patch_image(x);

    % Define the backward operator
    Pt = @(x) unpatch_image(x);

end
