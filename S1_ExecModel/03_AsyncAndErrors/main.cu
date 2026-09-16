// 03_AsyncAndErrors — S1: 커널은 비동기로 런치된다 + 런치 에러는 어떻게 잡나
//
// A) 비동기 런치
//    무거운 커널을 런치하고
//      (1) 런치 직후 CPU 스톱워치         → 거의 0 ms (런치 비용만 잼)
//      (2) cudaDeviceSynchronize 후 CPU  → 실제 실행 시간
//      (3) GpuTimer(cudaEvent)            → 실제 실행 시간 (동기화 없이 GPU 쪽에서 잼)
//    그리고 런치 직후 CPU 가 "다른 일"을 할 수 있다는 것도 확인한다.
//
// B) 런치 에러
//    block = 2048 (최대 1024 초과) 로 런치 → 아무 일도 안 일어난 것처럼 보인다.
//    cudaPeekAtLastError / cudaGetLastError 로 꺼내보고, Get 이 에러를 "지운다"는 것을 확인.
//
// C) 실행 에러는 여기서 일부러 내지 않는다.
//    경계 밖 접근(illegal address)은 sticky 에러라 컨텍스트가 망가져 이후 모든 호출이 실패한다.
//    → S4 에서 compute-sanitizer 로 안전하게 다룬다.

#include "cu_common.h"

#include <cstdio>
#include <thread>

// ---------------------------------------------------------------- 커널
// 원소마다 반복 연산을 해서 일부러 무겁게 만든 커널
__global__ void HeavyWork(float* data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = data[i];
    for (int k = 0; k < iters; ++k) {
        v = v * 0.999f + 0.5f / (1.0f + v * v);
    }
    data[i] = v;
}

__global__ void Nop(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1;
}

static void PartA_AsyncLaunch() {
    std::printf("[A] 비동기 런치\n");

    const int n = 10'000'000;
    const int iters = 100;   // 너무 키우면 Windows TDR(2초)로 드라이버가 리셋된다
    const int block = 256;
    const int grid = DivUp(n, block);

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));

    // 워밍업
    HeavyWork<<<grid, block>>>(d, n, 1);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer gpu;
    CpuTimer cpu;

    cpu.Start();
    gpu.Start();
    HeavyWork<<<grid, block>>>(d, n, iters);
    gpu.Stop();                                // 이벤트도 큐에 들어갈 뿐, 여기서 기다리지 않는다
    CUDA_CHECK_LAUNCH();
    const double afterLaunchMs = cpu.ElapsedMs();

    // GPU 가 일하는 동안 CPU 는 딴 일을 할 수 있다
    long long cpuSideWork = 0;
    for (int k = 0; k < 50'000'000; ++k) cpuSideWork += k & 7;
    const double afterCpuWorkMs = cpu.ElapsedMs();

    CUDA_CHECK(cudaDeviceSynchronize());
    const double afterSyncMs = cpu.ElapsedMs();
    const double gpuMs = gpu.ElapsedMs();

    std::printf("  (1) 런치 직후 CPU 시간            : %9.3f ms   ← 이걸 커널 시간이라고 착각하면 안 됨\n", afterLaunchMs);
    std::printf("      (그 사이 CPU 작업 끝난 시각)   : %9.3f ms   (sum=%lld)\n", afterCpuWorkMs, cpuSideWork);
    std::printf("  (2) cudaDeviceSynchronize 후 CPU  : %9.3f ms\n", afterSyncMs);
    std::printf("  (3) GpuTimer (cudaEvent)          : %9.3f ms\n", gpuMs);
    std::printf("  → (2) ≈ max(CPU 작업, GPU 작업) 이면 CPU·GPU 가 겹쳐 돈 것. 이게 S6~S7 의 출발점.\n\n");

    CUDA_CHECK(cudaFree(d));
}

static void PartB_LaunchError() {
    std::printf("[B] 런치 에러 (block 2048 > 1024)\n");

    const int n = 4096;
    int* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(int)));

    Nop<<<2, 2048>>>(d, n);                   // 잘못된 런치 — 반환값이 없어서 조용하다
    std::printf("  런치 직후: 아무 출력 없음 (에러가 조용히 저장됨)\n");

    cudaError_t e1 = cudaPeekAtLastError();   // 보기만 한다
    cudaError_t e2 = cudaPeekAtLastError();   // 여전히 남아 있다
    cudaError_t e3 = cudaGetLastError();      // 꺼내면서 지운다
    cudaError_t e4 = cudaGetLastError();      // 이제 success

    std::printf("  Peek #1 : %s\n", cudaGetErrorName(e1));
    std::printf("  Peek #2 : %s\n", cudaGetErrorName(e2));
    std::printf("  Get  #1 : %s  (%s)\n", cudaGetErrorName(e3), cudaGetErrorString(e3));
    std::printf("  Get  #2 : %s\n", cudaGetErrorName(e4));

    // 런치 설정 에러는 sticky 가 아니다 → 컨텍스트는 멀쩡, 계속 쓸 수 있다
    Nop<<<DivUp(n, 1024), 1024>>>(d, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("  올바른 런치(block 1024)는 정상 동작 → 컨텍스트 살아 있음\n\n");

    CUDA_CHECK(cudaFree(d));
}

int main() {
    std::printf("=== 03_AsyncAndErrors ===\n\n");
    PartA_AsyncLaunch();
    PartB_LaunchError();

    std::printf("[정리]\n");
    std::printf("  - 커널 시간은 CPU 스톱워치가 아니라 cudaEvent 로 잰다 (또는 동기화 후 CPU 로).\n");
    std::printf("  - 런치 뒤엔 CUDA_CHECK_LAUNCH() — 안 하면 에러가 다음 API 호출에 엉뚱하게 튀어나온다.\n");
    std::printf("  - 실행 중 에러(경계 밖 접근)는 sticky → S4 compute-sanitizer.\n");
    return 0;
}
