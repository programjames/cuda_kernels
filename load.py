import torch
from torch.utils.cpp_extension import load

# Compile and load the CUDA kernel
custom_kernel = load(
    name="kernel",
    sources=["kernel.cu"],
    verbose=True
)

def test(a, b):
    # Ensure the tensors are on the GPU
    a = a.contiguous().cuda()
    b = b.contiguous().cuda()
    
    # Prepare the output tensor
    c = torch.empty_like(a)
    
    # Launch the kernel
    threads_per_block = 1024
    blocks_per_grid = (a.numel() + threads_per_block - 1) // threads_per_block
    custom_kernel.kernel(blocks_per_grid, threads_per_block, a.data_ptr(), b.data_ptr(), c.data_ptr(), a.numel())
    
    return c
