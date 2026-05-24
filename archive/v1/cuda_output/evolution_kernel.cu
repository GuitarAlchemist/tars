
__global__ void evolution_kernel(float* data, int n, float evolution_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * evolution_factor + 0.1f;
    }
}
