#include <torch/extension.h>

__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int size = a.size(0);
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);
}