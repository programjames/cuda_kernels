extern "C" __global__ void multiply_add_kernel(float* a, float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n / 2) {
        // Multiply for the first half
        c[index] = a[index] * b[index];
    } else if (index < n) {
        // Add for the second half
        c[index] = a[index] + b[index];
    }
}