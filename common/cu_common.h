// cu_common.h — 모든 예제가 같이 쓰는 최소 유틸
//   CUDA_CHECK(call)     : CUDA API 반환값 검사. 실패하면 이름/설명/위치 찍고 종료
//   CUDA_CHECK_LAUNCH()  : 커널 런치 직후 호출. 런치 설정 에러는 cudaGetLastError로만 잡힌다
//   DivUp(a, b)          : 올림 나눗셈 (grid 크기 계산)
//   GpuTimer / CpuTimer  : GPU 이벤트 기반 시간 / CPU 벽시계 시간
//   Median(v)            : 반복 측정 중앙값
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------- 에러 처리
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t cc_status_ = (call);                                        \
        if (cc_status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "\n[CUDA ERROR] %s\n  -> %s: %s\n  at %s:%d\n", \
                         #call, cudaGetErrorName(cc_status_),                   \
                         cudaGetErrorString(cc_status_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 커널 런치는 반환값이 없다. 잘못된 grid/block 같은 "런치 에러"는
// 여기서 cudaGetLastError()로 꺼내야 보인다(꺼내면 지워진다).
// 커널 "실행 중" 에러(경계 밖 접근 등)는 비동기라 다음 동기화 지점에서 나온다.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// ---------------------------------------------------------------- 런치 계산
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------- 시간 측정
// GPU 타이머: 기본 스트림(0)에 이벤트를 찍고, 끝 이벤트가 완료될 때까지 기다린 뒤 차이를 잰다.
// 커널은 비동기라 CPU 스톱워치로 런치 전후를 재면 "런치 비용"만 재게 된다 → 03 예제 참고.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&end_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(end_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void Start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    void Stop() { CUDA_CHECK(cudaEventRecord(end_, 0)); }

    // 내부에서 end 이벤트 동기화까지 하므로 Stop() 뒤 바로 불러도 된다.
    double ElapsedMs() {
        CUDA_CHECK(cudaEventSynchronize(end_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, end_));
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t end_{};
};

class CpuTimer {
public:
    using Clock = std::chrono::steady_clock;
    void Start() { t0_ = Clock::now(); }
    double ElapsedMs() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    }

private:
    Clock::time_point t0_ = Clock::now();
};

inline double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
