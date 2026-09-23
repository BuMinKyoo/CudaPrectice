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
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지
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

    // ---- ① 이 PC에 CUDA를 쓸 수 있는 GPU가 몇 개인지 센다 (NVIDIA GPU만 셈)
    int count = 0;
    // GPU가 없거나 드라이버가 없으면 여기서 걸러져 프로그램이 끝난다
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::printf("\n[Devices] %d\n", count);

    // GPU를 하나씩 돌며 "무엇인지"(스펙)와 "실제로 쓸 수 있는지"(smoke test)를 함께 확인한다
    for (int dev = 0; dev < count; ++dev) {
        // ---- ② 현재 디바이스 지정. 이후의 할당/런치/MemGetInfo가 모두 이 GPU에 적용된다
        //      (첫 호출 때 컨텍스트가 만들어진다 → 수십~수백 ms + VRAM 일부 소비)
        CUDA_CHECK(cudaSetDevice(dev));

        // ---- ③ 스펙 조회. {} 로 0 초기화해서 쓰레기 값이 남지 않게 한다
        cudaDeviceProp props{};   // cuda_runtime_api.h 의 struct cudaDeviceProp
        // dev 를 인자로 받는다 → 현재 디바이스와 무관하게 조회 가능 (변하지 않는 카탈로그 정보)
        CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

        size_t freeB = 0, totalB = 0;
        // 반대로 이쪽은 GPU 번호 인자가 없다 → 현재 디바이스 전용 (지금 이 순간의 상태)
        // free < total 은 정상: 화면 출력·다른 앱·내 컨텍스트가 VRAM을 쓴다
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

        // ---- 출력값은 전부 앞으로 커널을 짤 때 제약 조건이 되는 값들이다
        std::printf("\n  #%d %s\n", dev, props.name);
        // major, minor 를 붙여서 sm_89 를 만든다 (sm_8.9 가 아니다) → 빌드 옵션에 그대로 들어감
        std::printf("    Compute Capability : %d.%d   → Directory.Build.props CudaArch = sm_%d%d\n",
                    props.major, props.minor, props.major, props.minor);
        // 1024.0 처럼 실수로 나눈다. 정수로 나누면 소수점이 잘린다
        std::printf("    VRAM               : %.2f GB total / %.2f GB free\n",
                    totalB / (1024.0 * 1024.0 * 1024.0), freeB / (1024.0 * 1024.0 * 1024.0));
        // 32. 블록 크기를 32의 배수로 잡는 근거 (→ 01 블록 크기 실험)
        std::printf("    warp size          : %d threads\n", props.warpSize);
        // 1024. 넘기면 런치 에러 (→ 03)
        std::printf("    max threads/block  : %d\n", props.maxThreadsPerBlock);
        std::printf("    max grid size      : %d x %d x %d\n",
                    props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        // size_t 이므로 %d 가 아니라 %zu. S2에서 타일 크기를 정하는 기준이 된다
        std::printf("    shared mem/block   : %zu bytes (%.1f KB)\n",
                    props.sharedMemPerBlock, props.sharedMemPerBlock / 1024.0);
        // occupancy 계산에 쓰인다
        std::printf("    registers/block    : %d\n", props.regsPerBlock);
        // 동시에 일하는 "작은 프로세서" 수. 스레드를 얼마나 띄워야 GPU가 다 차는지의 기준
        std::printf("    SM count           : %d\n", props.multiProcessorCount);
        // 화면을 그리는 GPU면 ON → 커널이 2초를 넘으면 Windows가 GPU를 리셋한다
        std::printf("    kernel timeout(TDR): %s\n", props.kernelExecTimeoutEnabled ? "ON (커널 2초 제한)" : "OFF");

        // ---- ④ 실제 동작 검증. printf 인자가 먼저 평가되므로 테스트가 끝난 뒤 결과가 찍힌다
        //      현재 디바이스에서 돌기 때문에 GPU가 여러 개면 각각 따로 검증된다
        std::printf("    smoke test         : %s\n", SmokeTest() ? "OK" : "FAIL");
    }

    // 여기서 얻은 숫자가 S1 이후 모든 실험의 기준값이 된다
    std::printf("\n→ 위 값들을 README.md '내 환경' 표에 옮겨 적는다.\n");
    return 0;
}
