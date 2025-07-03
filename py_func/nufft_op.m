function k_samples = nufft_op(im)
    % function k_samples = nufft_op(im)
    % NUFFT operator
    % Input:
    % im - image [Nx, Ny, L]
    % Output:
    % k_samples - k-space samples [L, 1, num_samples]

    tmp = py.py_func.nufft_for_matlab.nufft_forward_op(py.numpy.array(im));
    k_samples = double(tmp{'k_samples'});
end
