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
        // 이 GB 는 GPU 칩 "바깥" 기판 위의 GDDR 칩(전역 메모리)만 센 값이다.
        // 아래 [SM 하나의 수용량] 의 shared mem 은 칩 "안" SRAM 이라 여기 포함되지 않는다.
        //   VRAM        칩 바깥 DRAM   ~500 사이클   수 GB      <- cudaMalloc 이 잡는 곳
        //   공유 메모리  칩 안   SRAM   ~30 사이클   SM당 수십 KB <- __shared__ 가 쓰는 곳
        // 종류가 다른 반도체라 속도가 수십 배 차이 난다. cudaMalloc 으로 VRAM 을 꽉 채워도
        // 각 SM 의 공유 메모리는 그대로 남아 있다 (별개의 물리 메모리).
        std::printf("    VRAM               : %.2f GB total / %.2f GB free\n",
                    totalB / (1024.0 * 1024.0 * 1024.0), freeB / (1024.0 * 1024.0 * 1024.0));
        // 32. 블록 크기를 32의 배수로 잡는 근거 (→ 01 블록 크기 실험)
        std::printf("    warp size          : %d threads\n", props.warpSize);
        // 동시에 일하는 "작은 프로세서" 수. 스레드를 얼마나 띄워야 GPU가 다 차는지의 기준
        std::printf("    SM count           : %d\n", props.multiProcessorCount);
        // 화면을 그리는 GPU면 ON → 커널이 2초를 넘으면 Windows가 GPU를 리셋한다
        std::printf("    kernel timeout(TDR): %s\n", props.kernelExecTimeoutEnabled ? "ON (커널 2초 제한)" : "OFF");

        // ---- 블록 하나가 넘을 수 없는 한계. 넘기면 런치 에러(→ 03)
        std::printf("\n    [블록 하나의 한계]\n");
        std::printf("    max threads/block  : %d\n", props.maxThreadsPerBlock);
        std::printf("    max grid size      : %d x %d x %d\n",
                    props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        // size_t 이므로 %d 가 아니라 %zu. S2에서 타일 크기를 정하는 기준이 된다
        // 커널 안 __shared__ 선언을 다 합쳐서 넘을 수 없는 양. 넘기면 런치 에러(→ 03).
        //   예) __shared__ float tile[32][33] = 32*33*4 = 4,224 바이트
        // 아래 shared mem/SM 과 숫자가 다른 이유는 거기 주석 참고.
        std::printf("    shared mem/block   : %zu bytes (%.1f KB)\n",
                    props.sharedMemPerBlock, props.sharedMemPerBlock / 1024.0);
        std::printf("    registers/block    : %d\n", props.regsPerBlock);

        // ---- SM 하나가 "동시에" 품을 수 있는 양. 위의 /block 값과 혼동하지 말 것.
        //      블록이 SM에 올라오면 아래 자리를 동시에 차지하고, 먼저 바닥나는 쪽이 한계가 된다.
        //      예) block 256 → 스레드 자리에 먼저 걸림: 1536/256 = 블록 6개 (정원 100%)
        //          block  32 → 블록 자리에 먼저 걸림:  24개뿐 → 768명     (정원  50%)
        std::printf("\n    [SM 하나의 수용량]\n");
        // 정원. 코어 수(128 등)와는 다른 개념 — 동시에 "머무는" 수이지 동시에 "계산하는" 수가 아니다
        std::printf("    max threads/SM     : %d  (warp %d개분)\n",
                    props.maxThreadsPerMultiProcessor,
                    props.maxThreadsPerMultiProcessor / props.warpSize);
        // SM 이 블록별 관리 정보(blockIdx, shared 영역, __syncthreads 상태)를 담아 두는 칸 수
        std::printf("    max blocks/SM      : %d\n", props.maxBlocksPerMultiProcessor);
        // 위 정원의 근거. 스레드마다 자기 레지스터를 미리 배정받기 때문에 전환 비용이 0 이다
        std::printf("    registers/SM       : %d  (스레드당 약 %d개)\n",
                    props.regsPerMultiprocessor,
                    props.regsPerMultiprocessor / props.maxThreadsPerMultiProcessor);
        // /block 과 /SM 두 숫자가 따로 있는 이유 — 물리적으로는 SM 당 하나지만,
        // 블록마다 칸을 나눠 주기 때문이다. 블록이 SM 에 올라올 때 자기 몫을 떼어 받고,
        // 끝나면 반납해서 다음 블록이 그 자리를 쓴다.
        //   ┌──── SM 의 공유 메모리 (물리적으로 하나) ────┐
        //   │ [블록A][블록B][블록C][블록D]    (빈 공간)   │
        //   │   A만    B만    C만    D만                 │
        //   └────────────────────────────────────────────┘
        //            서로 넘볼 수 없다 (칸막이)
        //
        // 블록 간에는 공유할 수 없다. 이유 셋:
        //   - 동시에 상주한다는 보장이 없다 (블록 0 과 블록 15000 은 만날 일이 없다)
        //   - 어느 SM 에 배정될지 모른다 (다른 칩 구역이면 공유 자체가 불가능)
        //   - 같은 바이너리가 SM 8개 GPU 와 132개 GPU 에서 모두 돌아야 한다
        // 블록 간 통신은 전역 메모리로, 동기화는 커널을 나누는 것으로 한다
        // (커널 경계 = 모든 블록이 끝난 지점 = 사실상 유일한 전역 동기화 수단).
        //
        // 그리고 이 값이 occupancy 제약에 하나 더 추가된다:
        //   ① 블록 자리        max blocks/SM
        //   ② warp/스레드 자리  max threads/SM
        //   ③ 공유 메모리      shared mem/SM ÷ 블록당 사용량     <- 여기
        //   ④ 레지스터         registers/SM ÷ 블록당 사용량
        //   SM 당 블록 수 = 위 넷 중 제일 작은 값
        // 블록당 한도를 꽉 쓰면 (/SM ÷ /block) 개밖에 안 올라간다 → 공유 메모리가
        // occupancy 를 직접 깎는 상황이 된다. 실제 예는 04_Transpose.
        std::printf("    shared mem/SM      : %zu bytes (%.1f KB)\n",
                    props.sharedMemPerMultiprocessor, props.sharedMemPerMultiprocessor / 1024.0);
        // 이 GPU 전체가 동시에 품는 스레드 수. 이만큼은 띄워야 GPU 가 꽉 찬다
        std::printf("    → GPU 전체 동시 상주 : %d threads (%d SM x %d)\n",
                    props.maxThreadsPerMultiProcessor * props.multiProcessorCount,
                    props.multiProcessorCount, props.maxThreadsPerMultiProcessor);

        // ---- ④ 실제 동작 검증. printf 인자가 먼저 평가되므로 테스트가 끝난 뒤 결과가 찍힌다
        //      현재 디바이스에서 돌기 때문에 GPU가 여러 개면 각각 따로 검증된다
        std::printf("    smoke test         : %s\n", SmokeTest() ? "OK" : "FAIL");
    }

    // 여기서 얻은 숫자가 S1 이후 모든 실험의 기준값이 된다
    std::printf("\n→ 위 값들을 README.md '내 환경' 표에 옮겨 적는다.\n");
    return 0;
}
