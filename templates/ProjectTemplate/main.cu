// __PROJECT_NAME__ — (한 줄 설명)
//
// 배울 것:
//   -
// 실험:
//   A)

#include "cu_common.h"

#include <cstdio>
#include <vector>

__global__ void MyKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;
    }
}

int main() {
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지
    std::printf("=== __PROJECT_NAME__ ===\n\n");

    const int n = 1 << 20;
    const int block = 256;
    const size_t bytes = n * sizeof(float);

    std::vector<float> host(n, 1.0f);

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));

    GpuTimer t;
    t.Start();
    MyKernel<<<DivUp(n, block), block>>>(d, n);
    CUDA_CHECK_LAUNCH();
    t.Stop();

    CUDA_CHECK(cudaMemcpy(host.data(), d, bytes, cudaMemcpyDeviceToHost));
    std::printf("kernel %.3f ms, host[0] = %.1f\n", t.ElapsedMs(), host[0]);

    CUDA_CHECK(cudaFree(d));
    return 0;
}
