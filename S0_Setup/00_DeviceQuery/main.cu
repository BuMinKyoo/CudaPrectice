// 00_DeviceQuery — S0 환경 확인
//
// 확인하는 것
//   1) 드라이버가 지원하는 CUDA 버전 >= 런타임(툴킷) 버전인가
//   2) GPU 이름, Compute Capability(→ -arch=sm_XY), VRAM
//   3) 아주 작은 커널 하나가 실제로 돌아서 결과가 맞는가 (빌드·링크·런치 전 경로 확인)

#include "cu_common.h"

#include <cstdio>
#include <vector>

// ---------------------------------------------------------------- 커널
// 각 스레드가 자기 전역 인덱스를 써 넣는다. 결과가 0,1,2,...면 런치가 정상.
__global__ void WriteIndex(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

static void PrintVersion(const char* label, int v) {
    // CUDA 버전 정수 = 1000*major + 10*minor  (예: 12090 → 12.9)
    std::printf("  %-22s %d.%d\n", label, v / 1000, (v % 1000) / 10);
}

static bool SmokeTest() {
    const int n = 1000;
    const int block = 128;
    const int grid = DivUp(n, block);

    int* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(int)));

    WriteIndex<<<grid, block>>>(d, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h(n, -1);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    for (int i = 0; i < n; ++i) {
        if (h[i] != i) {
            std::printf("  smoke test FAIL at %d (got %d)\n", i, h[i]);
            return false;
        }
    }
    return true;
}

int main() {
    std::printf("=== 00_DeviceQuery ===\n\n");

    int drv = 0, rt = 0;
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    std::printf("[Version]\n");
    PrintVersion("Driver supports CUDA", drv);
    PrintVersion("Runtime (toolkit)", rt);
    if (rt > drv) {
        std::printf("  !! 런타임이 드라이버보다 새 버전 → 드라이버 업데이트 필요\n");
    }

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::printf("\n[Devices] %d\n", count);

    for (int dev = 0; dev < count; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));

        cudaDeviceProp props{};   // cuda_runtime_api.h 의 struct cudaDeviceProp
        CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

        size_t freeB = 0, totalB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

        std::printf("\n  #%d %s\n", dev, props.name);
        std::printf("    Compute Capability : %d.%d   → Directory.Build.props CudaArch = sm_%d%d\n",
                    props.major, props.minor, props.major, props.minor);
        std::printf("    VRAM               : %.2f GB total / %.2f GB free\n",
                    totalB / (1024.0 * 1024.0 * 1024.0), freeB / (1024.0 * 1024.0 * 1024.0));
        std::printf("    warp size          : %d threads\n", props.warpSize);
        std::printf("    max threads/block  : %d\n", props.maxThreadsPerBlock);
        std::printf("    max grid size      : %d x %d x %d\n",
                    props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        std::printf("    shared mem/block   : %zu bytes (%.1f KB)\n",
                    props.sharedMemPerBlock, props.sharedMemPerBlock / 1024.0);
        std::printf("    registers/block    : %d\n", props.regsPerBlock);
        std::printf("    SM count           : %d\n", props.multiProcessorCount);
        std::printf("    kernel timeout(TDR): %s\n", props.kernelExecTimeoutEnabled ? "ON (커널 2초 제한)" : "OFF");

        std::printf("    smoke test         : %s\n", SmokeTest() ? "OK" : "FAIL");
    }

    std::printf("\n→ 위 값들을 README.md '내 환경' 표에 옮겨 적는다.\n");
    return 0;
}
