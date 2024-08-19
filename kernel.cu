#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace
{
    __global__ void _test_kernel(float *a, float *b, float *c, int n)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < n / 2)
        {
            // Multiply for the first half
            c[index] = a[index] * b[index];
        }
        else if (index < n)
        {
            // Add for the second half
            c[index] = a[index] + b[index];
        }
    }
}

size_t div_round_up(size_t x, size_t y)
{
    return (x + y - 1) / y;
}

void test_wrapper(const at::Tensor in_a,
                  const at::Tensor in_b,
                  at::Tensor out_c,
                  int block_size = 64)
{
    size_t N = in_a.numel();
    size_t num_blocks = div_round_up(N, block_size);
    _test_kernel<<<num_blocks, block_size>>>(
        in_a.data_ptr<float>(), in_b.data_ptr<float>(), out_c.data_ptr<float>(), N);
}