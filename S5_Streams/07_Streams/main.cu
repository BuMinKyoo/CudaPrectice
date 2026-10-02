// 07_Streams — S5 전송과 겹치기: 03 이 끝내지 못한 이야기
//
// 03 은 "커널 런치는 비동기다" 까지 가르치고 끝났다. 그런데 정작 쓸 때는
// 매번 cudaDeviceSynchronize() 를 불렀다. 비동기를 배웠는데 동기적으로만 쓴 셈이다.
// 여기서 그걸 닫는다.
//
// 그리고 01 에서 확인한 숙제가 하나 더 있다.
//   커널 0.5 ms  vs  H2D 6.6 ms + D2H 3.7 ms
//   -> 전체의 95% 가 복사였고, GPU 가 CPU 보다 3배 느렸다
// 이 예제는 그 복사 시간을 두 가지 방법으로 줄인다.
//   1) pinned memory 로 복사 자체를 빠르게
//   2) 복사와 커널을 겹쳐서 돌려 "따로 걸리는 시간" 을 없애기
//
// 네 가지 방식 (하는 일은 전부 같다: H2D -> 커널 -> D2H)
//   A) pageable + 동기 복사      01 에서 쓰던 방식. 기준선
//   B) pinned   + 동기 복사      복사만 빨라진다
//   C) pinned   + 비동기 + 스트림 1개   겹칠 상대가 없어서 B 와 비슷하다
//   D) pinned   + 비동기 + 스트림 N개   드디어 겹친다
//
// 사용법:  07_Streams.exe [MB] [iters] [streams]
//          기본 128 MB, iters 200, streams 4

#include "cu_common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------- 커널
// 겹치는 것을 보려면 커널 시간이 복사 시간과 비슷해야 한다.
// 커널이 너무 짧으면 복사가 다 차지해서 겹쳐도 티가 안 난다.
// iters 로 작업량을 조절한다 (기본값은 복사와 비슷해지도록 잡아둔 값).
__global__ void Work(float* data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = data[i];
    for (int k = 0; k < iters; ++k) {
        v = v * 0.999f + 0.5f / (1.0f + v * v);
    }
    data[i] = v;
}

// ---------------------------------------------------------------- 헬퍼
static double BandwidthGBs(size_t bytes, double ms) {
    return static_cast<double>(bytes) / (ms * 1.0e6);
}

int main(int argc, char** argv) {
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지

    const int totalMB = (argc > 1) ? std::atoi(argv[1]) : 128;
    // iters 는 커널 시간을 전송 시간과 비슷하게 맞추려고 고른 값이다.
    // 커널이 너무 무거우면 겹쳐도 비율이 안 보인다 (전송이 다 숨어도 커널 시간은 그대로다).
    // 반대로 너무 가벼우면 전송이 다 차지해서 역시 안 보인다.
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 50;
    const int nStreams = (argc > 3) ? std::atoi(argv[3]) : 4;
    const int nChunks = (argc > 4) ? std::atoi(argv[4]) : 16;   // 조각 수. 스트림보다 충분히 많아야 한다

    const size_t totalBytes = static_cast<size_t>(totalMB) * 1024 * 1024;
    const int n = static_cast<int>(totalBytes / sizeof(float));
    const int chunkN = n / nChunks;
    const size_t chunkBytes = static_cast<size_t>(chunkN) * sizeof(float);

    const int block = 256;

    std::printf("=== 07_Streams  %d MB (%d floats), iters = %d ===\n\n", totalMB, n, iters);
    // 주의: 한글로 끝낸 뒤 \n 을 쓰면 안 된다 (nvcc 가 /utf-8 을 못 받아 삼킨다).
    std::printf("조각 %d개 x %.1f MB, 스트림 %d\n\n",
                nChunks, chunkBytes / 1048576.0, nStreams);

    // ---- 이 GPU 가 겹치기를 지원하는가
    // asyncEngineCount = 복사 엔진 수.
    //   0 : 복사와 커널을 겹칠 수 없다
    //   1 : 한 방향 복사와 커널을 겹칠 수 있다
    //   2 : 양방향 복사(H2D, D2H)와 커널을 동시에 겹칠 수 있다
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
    std::printf("[%s] asyncEngineCount = %d, concurrentKernels = %d\n",
                props.name, props.asyncEngineCount, props.concurrentKernels);
    if (props.asyncEngineCount == 0) {
        std::printf("  !! 이 GPU 는 복사와 커널을 겹칠 수 없다. D 가 C 와 비슷하게 나올 것이다.\n");
    } else if (props.asyncEngineCount == 1) {
        std::printf("  복사 엔진 1개. 한 방향씩만 겹친다 (H2D 와 커널, 또는 D2H 와 커널).\n");
    } else {
        std::printf("  복사 엔진 2개. H2D, 커널, D2H 셋을 동시에 겹칠 수 있다.\n");
    }
    std::printf("\n");

    // ---- 호스트 메모리 두 종류
    // pageable : 일반 malloc/new. OS 가 언제든 디스크로 밀어낼 수 있다
    //            -> GPU 가 직접 못 읽는다. 드라이버가 내부 pinned 버퍼로 한 번 복사한 뒤
    //               거기서 GPU 로 보낸다. 복사가 두 번이다.
    // pinned   : cudaMallocHost. OS 가 밀어내지 못하게 고정(page-locked)한다
    //            -> GPU 가 바로 읽는다. 복사 한 번. 그리고 비동기 복사가 가능해진다.
    std::vector<float> hPageable(n);
    for (int i = 0; i < n; ++i) {
        hPageable[i] = static_cast<float>(i % 1000) * 0.001f;
    }

    float* hPinned = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&hPinned, totalBytes));   // <- pinned 할당
    for (int i = 0; i < n; ++i) {
        hPinned[i] = hPageable[i];
    }

    float* dFull = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dFull, totalBytes));

    CpuTimer t;

    // ================================================================
    // 먼저 복사 속도만 따로 잰다 (pinned 효과를 분리해서 보기 위해)
    // ================================================================
    std::printf("[전송 속도만] %d MB 한 번 보내는 데 걸리는 시간\n", totalMB);

    CUDA_CHECK(cudaMemcpy(dFull, hPageable.data(), totalBytes, cudaMemcpyHostToDevice)); // 워밍업
    t.Start();
    CUDA_CHECK(cudaMemcpy(dFull, hPageable.data(), totalBytes, cudaMemcpyHostToDevice));
    const double h2dPageable = t.ElapsedMs();

    t.Start();
    CUDA_CHECK(cudaMemcpy(dFull, hPinned, totalBytes, cudaMemcpyHostToDevice));
    const double h2dPinned = t.ElapsedMs();

    std::printf("    pageable : %7.2f ms  (%5.1f GB/s)\n",
                h2dPageable, BandwidthGBs(totalBytes, h2dPageable));
    std::printf("    pinned   : %7.2f ms  (%5.1f GB/s)   -> %.2fx\n\n",
                h2dPinned, BandwidthGBs(totalBytes, h2dPinned), h2dPageable / h2dPinned);

    // ---- 구성 요소를 따로 재둔다. 겹침 결과를 읽으려면 "숨길 수 있는 양" 을 알아야 한다.
    float* hTmp = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&hTmp, totalBytes));
    t.Start();
    CUDA_CHECK(cudaMemcpy(hTmp, dFull, totalBytes, cudaMemcpyDeviceToHost));
    const double d2hPinned = t.ElapsedMs();

    Work<<<DivUp(n, block), block>>>(dFull, n, iters);     // 워밍업
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    t.Start();
    Work<<<DivUp(n, block), block>>>(dFull, n, iters);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    const double kernelMs = t.ElapsedMs();
    CUDA_CHECK(cudaFreeHost(hTmp));

    std::printf("[구성 요소] 따로따로 재면\n");
    std::printf("    H2D (pinned) : %7.2f ms\n", h2dPinned);
    std::printf("    커널          : %7.2f ms\n", kernelMs);
    std::printf("    D2H (pinned) : %7.2f ms\n", d2hPinned);
    std::printf("    단순 합       : %7.2f ms   <- 안 겹치면 이만큼 (serial)\n",
                h2dPinned + kernelMs + d2hPinned);
    std::printf("    커널만        : %7.2f ms   <- 복사를 다 숨기면 여기까지 (floor)\n\n",
                kernelMs);

    // ================================================================
    // A) pageable + 동기 복사 — 01 에서 쓰던 방식
    // ================================================================
    auto runSync = [&](const float* hSrc, float* hDst) {
        CUDA_CHECK(cudaMemcpy(dFull, hSrc, totalBytes, cudaMemcpyHostToDevice));
        Work<<<DivUp(n, block), block>>>(dFull, n, iters);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(hDst, dFull, totalBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    std::vector<float> hOutPageable(n);
    runSync(hPageable.data(), hOutPageable.data());   // 워밍업

    t.Start();
    runSync(hPageable.data(), hOutPageable.data());
    const double msA = t.ElapsedMs();

    // ================================================================
    // B) pinned + 동기 복사 — 복사만 빨라진다. 구조는 A 와 동일
    // ================================================================
    float* hOutPinned = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&hOutPinned, totalBytes));

    t.Start();
    runSync(hPinned, hOutPinned);
    const double msB = t.ElapsedMs();

    // ================================================================
    // C) / D) pinned + 비동기 + 스트림
    //
    // 핵심은 "조각으로 잘라서 파이프라인처럼 흘려보내는 것" 이다.
    //   조각0: [H2D][커널][D2H]
    //   조각1:      [H2D][커널][D2H]       <- 조각0 이 커널 도는 동안 조각1 을 올린다
    //   조각2:           [H2D][커널][D2H]
    //
    // cudaMemcpyAsync 는 pinned 메모리에서만 진짜로 비동기다.
    // pageable 로 부르면 드라이버가 내부에서 동기 복사로 처리해 버린다 (조용히).
    // ================================================================
    // 스트림은 측정 "밖에서" 미리 만든다. cudaStreamCreate/Destroy 는 공짜가 아니고,
    // Destroy 는 그 스트림의 작업이 끝나기를 기다리기도 한다. 측정 구간에 넣으면
    // 그 비용이 겹침 효과를 가려버린다.
    std::vector<cudaStream_t> st(nStreams);
    for (int s = 0; s < nStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&st[s]));
    }

    auto runAsync = [&](int streams) {
        for (int c = 0; c < nChunks; ++c) {
            const int s = c % streams;
            const size_t off = static_cast<size_t>(c) * chunkN;

            CUDA_CHECK(cudaMemcpyAsync(dFull + off, hPinned + off, chunkBytes,
                                       cudaMemcpyHostToDevice, st[s]));
            Work<<<DivUp(chunkN, block), block, 0, st[s]>>>(dFull + off, chunkN, iters);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpyAsync(hOutPinned + off, dFull + off, chunkBytes,
                                       cudaMemcpyDeviceToHost, st[s]));
        }

        CUDA_CHECK(cudaDeviceSynchronize());   // 전부 끝날 때까지 여기서 한 번만 기다린다
    };

    runAsync(nStreams);          // 워밍업

    t.Start();
    runAsync(1);
    const double msC = t.ElapsedMs();

    t.Start();
    runAsync(nStreams);
    const double msD = t.ElapsedMs();

    // ================================================================
    // 결과
    // ================================================================
    std::printf("[전체 파이프라인] H2D + 커널 + D2H\n\n");
    std::printf("| 방식 | 총 시간 | A 대비 |\n");
    std::printf("|---|---:|---:|\n");
    std::printf("| A) pageable + 동기        | %7.2f ms | 1.00x |\n", msA);
    std::printf("| B) pinned   + 동기        | %7.2f ms | %.2fx |\n", msB, msA / msB);
    std::printf("| C) pinned   + 비동기 1스트림 | %7.2f ms | %.2fx |\n", msC, msA / msC);
    std::printf("| D) pinned   + 비동기 %dstream | %7.2f ms | %.2fx |\n", nStreams, msD, msA / msD);

    // 검증 — 네 방식 모두 같은 일을 했으므로 결과가 같아야 한다
    size_t bad = 0;
    for (int i = 0; i < n; ++i) {
        if (hOutPinned[i] != hOutPageable[i]) {
            ++bad;
        }
    }
    std::printf("\n    mismatch (A vs D) = %zu\n", bad);

    const double floorMs = kernelMs;
    const double serialMs = h2dPinned + kernelMs + d2hPinned;
    std::printf("\n    단순 합 %.2f ms -> D %.2f ms,  커널만 돌릴 때는 %.2f ms\n",
                serialMs, msD, floorMs);
    std::printf("    숨긴 복사 시간 = %.2f ms  (숨길 수 있던 양 %.2f ms 중 %.0f%%)\n",
                serialMs - msD, serialMs - floorMs,
                100.0 * (serialMs - msD) / (serialMs - floorMs));

    std::printf("\n[읽는 법]\n");
    std::printf("  A -> B  복사 자체가 빨라진 효과. 구조는 똑같고 메모리 종류만 바꿨다.\n");
    std::printf("  B -> C  거의 차이가 없어야 한다. 스트림이 1개면 겹칠 상대가 없다.\n");
    std::printf("          '비동기로 부르는 것' 과 '실제로 겹치는 것' 은 다른 이야기다.\n");
    std::printf("  C -> D  여기서 겹친다. 조각들이 서로 다른 스트림에 들어가므로,\n");
    std::printf("          앞 조각이 커널을 도는 동안 뒤 조각을 올린다.\n");
    std::printf("  D 가 '커널만' 에 가까울수록 복사를 완전히 숨긴 것이다.\n");

    std::printf("\n[겹침이 기대만큼 안 날 때 — 환경 탓인 경우가 많다]\n");
    std::printf("  1. asyncEngineCount 가 1 이면 복사 엔진이 하나다.\n");
    std::printf("     H2D 와 D2H 가 서로 겹치지 못하고 그 엔진에서 줄을 선다.\n");
    std::printf("     즉 이론 하한은 '커널만' 이 아니라 max(커널, H2D+D2H) 이다.\n");
    std::printf("       여기서는 max(%.1f, %.1f) = %.1f ms 가 현실적인 바닥이다.\n",
                kernelMs, h2dPinned + d2hPinned,
                kernelMs > (h2dPinned + d2hPinned) ? kernelMs : (h2dPinned + d2hPinned));
    std::printf("  2. Windows 의 WDDM 모드에서는 OS 가 GPU 스케줄링을 쥐고 있다.\n");
    std::printf("     드라이버가 명령을 묶어 보내서 잘게 겹치는 것이 잘 안 된다.\n");
    std::printf("     Linux 나 TCC 모드에서는 같은 코드가 훨씬 잘 겹친다.\n");
    std::printf("  3. 측정 편차도 크다. 여러 번 돌려서 경향을 보는 것이 맞다.\n");
    std::printf("\n  겹침이 실제로 일어났는지는 숫자보다 타임라인으로 보는 것이 확실하다:\n");
    std::printf("     nsys profile --stats=true 07_Streams.exe\n");

    for (int s = 0; s < nStreams; ++s) {
        CUDA_CHECK(cudaStreamDestroy(st[s]));
    }
    CUDA_CHECK(cudaFree(dFull));
    CUDA_CHECK(cudaFreeHost(hPinned));
    CUDA_CHECK(cudaFreeHost(hOutPinned));
    return 0;
}
