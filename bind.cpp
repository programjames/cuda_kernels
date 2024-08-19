
#include <torch/extension.h>

// forward declare public function from cuda file
void test_wrapper(const at::Tensor in_a, const at::Tensor in_b, at::Tensor out_c,
                 int block_size);

void test(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
         int block_size = 64) {
    test_wrapper(in_a, in_b, out_c, block_size);
}

PYBIND11_MODULE(kernel, m)
{
    m.doc() = "Custom CUDA kernel test";
    m.def("test", &test, "Custom kernel performing multiply-add on two tensors");
}