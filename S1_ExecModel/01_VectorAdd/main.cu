// 01_VectorAdd — S1 실행 모델: 블록 크기가 성능에 주는 영향 + grid-stride loop
//
// 실험
//   A) 블록 크기 8 ~ 1024 로 같은 벡터 덧셈을 돌려 커널 시간(중앙값) 비교
//      - 32 의 배수가 아닌 크기(100)도 일부러 넣었다 → warp 가 반만 찬다
//   B) grid-stride loop: 블록 수를 데이터 크기에 맞추지 않아도 전부 처리되는지
//
// 사용법:  01_VectorAdd.exe [N]      (기본 N = 10,000,000)

#include "cu_common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------- 커널
// 모놀리식: 스레드 1개 = 원소 1개. grid*block >= n 이어야 전부 처리된다.
__global__ void VecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                       // 반올림으로 남는 스레드는 아무것도 안 한다
        c[i] = a[i] + b[i];
    }
}

// grid-stride: 각 스레드가 "grid 전체 스레드 수" 간격으로 건너뛰며 여러 원소를 맡는다.
// 스레드가 n보다 적어도 빠짐없이 처리되고, 많아도 겹치지 않는다.
__global__ void VecAddGridStride(const float* a, const float* b, float* c, int n) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------- 헬퍼
static int CountMismatch(const std::vector<float>& ref, const std::vector<float>& got) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) ++bad;   // 덧셈은 CPU/GPU 모두 IEEE 754 → 비트 단위로 같아야 함
    }
    return bad;
}

int main(int argc, char** argv) {
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지
    const int n = (argc > 1) ? std::atoi(argv[1]) : 10'000'000;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int kRepeat = 15;

    std::printf("=== 01_VectorAdd  N = %d (%.1f MB per vector) ===\n\n", n, bytes / 1048576.0);

    // ---- 호스트 데이터
    std::vector<float> a(n), b(n), cCpu(n), cGpu(n);
    for (int i = 0; i < n; ++i) {
        a[i] = std::sin(static_cast<float>(i)) * 100.0f;
        b[i] = std::cos(static_cast<float>(i)) * 100.0f;
    }

    // ---- CPU 기준값 + 시간
    CpuTimer cpu;
    cpu.Start();
    for (int i = 0; i < n; ++i) cCpu[i] = a[i] + b[i];
    const double cpuMs = cpu.ElapsedMs();

    // ---- 디바이스 메모리
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dB, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dC, bytes));

    GpuTimer t;

    t.Start();
    CUDA_CHECK(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice));
    t.Stop();
    const double h2dMs = t.ElapsedMs();

    // ---- 워밍업 (첫 런치는 컨텍스트/JIT 비용이 섞인다)
    VecAdd<<<DivUp(n, 256), 256>>>(dA, dB, dC, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================== A) 블록 크기 스윕
    std::printf("[A] block size sweep (kernel only, median of %d)\n\n", kRepeat);
    std::printf("| block | grid | total threads | kernel ms | vs CPU | correct |\n");
    std::printf("|---:|---:|---:|---:|---:|:---:|\n");

    const int blockSizes[] = {8, 16, 32, 100, 128, 256, 512, 1024};
    for (int block : blockSizes) {
        const int grid = DivUp(n, block);
        std::vector<double> samples;
        samples.reserve(kRepeat);

        for (int r = 0; r < kRepeat; ++r) {
            t.Start();
            VecAdd<<<grid, block>>>(dA, dB, dC, n);
            CUDA_CHECK_LAUNCH();
            t.Stop();
            samples.push_back(t.ElapsedMs());
        }

        CUDA_CHECK(cudaMemcpy(cGpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
        const int bad = CountMismatch(cCpu, cGpu);
        const double ms = Median(samples);

        std::printf("| %4d | %8d | %10lld | %8.3f | %6.1fx | %s |\n",
                    block, grid, static_cast<long long>(grid) * block, ms,
                    cpuMs / ms, bad == 0 ? "OK" : "FAIL");
    }

    // =========================================================== B) grid-stride
    std::printf("\n[B] grid-stride loop (block 256)\n\n");
    std::printf("| grid | total threads | elems/thread | kernel ms | correct |\n");
    std::printf("|---:|---:|---:|---:|:---:|\n");

    const int gridsForStride[] = {1, 64, 4096, DivUp(n, 256)};
    for (int grid : gridsForStride) {
        std::fill(cGpu.begin(), cGpu.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(dC, cGpu.data(), bytes, cudaMemcpyHostToDevice)); // 이전 결과 지우기

        std::vector<double> samples;
        for (int r = 0; r < kRepeat; ++r) {
            t.Start();
            VecAddGridStride<<<grid, 256>>>(dA, dB, dC, n);
            CUDA_CHECK_LAUNCH();
            t.Stop();
            samples.push_back(t.ElapsedMs());
        }

        CUDA_CHECK(cudaMemcpy(cGpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
        const int bad = CountMismatch(cCpu, cGpu);
        const long long threads = static_cast<long long>(grid) * 256;

        std::printf("| %6d | %10lld | %8.1f | %9.3f | %s |\n",
                    grid, threads, static_cast<double>(n) / threads, Median(samples),
                    bad == 0 ? "OK" : "FAIL");
    }

    // ---- D2H 시간
    t.Start();
    CUDA_CHECK(cudaMemcpy(cGpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
    t.Stop();
    const double d2hMs = t.ElapsedMs();

    std::printf("\n[참고] CPU loop %.3f ms | H2D(2 vectors) %.3f ms | D2H %.3f ms\n",
                cpuMs, h2dMs, d2hMs);
    std::printf("       → 커널보다 복사가 훨씬 비싸다면: S5(pinned)에서 다시 본다.\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
