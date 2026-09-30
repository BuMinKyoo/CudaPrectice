// 05_BlurShared — S2 메모리: 공유 메모리로 "중복 읽기" 없애기 (halo/apron)
//
// 04 와 공유 메모리를 쓰는 이유가 다르다.
//   04_Transpose : 원소를 한 번씩만 읽는다. 공유 메모리는 접근 "순서"를 바꾸는 환승역이었다.
//   05_BlurShared: 원소를 이웃들이 여러 번 읽는다. 공유 메모리로 "중복"을 없앤다. ← 본래 목적
//
// 5x5 박스 블러는 출력 픽셀 하나에 입력 25개가 필요하다.
// 그런데 바로 옆 픽셀도 그 25개 중 20개를 똑같이 쓴다. 순진하게 짜면 같은 값을
// 전역 메모리에서 25번씩 읽는다. 타일로 한 번만 가져와 블록 안에서 돌려 쓰면 된다.
//
// 새로 나오는 개념: halo(apron)
//   타일 가장자리 픽셀을 계산하려면 타일 "밖" 이웃도 필요하다.
//   그래서 32x8 타일을 위해 (32+2R)x(8+2R) 만큼 읽어와야 한다.
//   즉 스레드 수보다 읽어야 할 칸이 더 많다 → 로딩 루프가 필요하다.
//
// 실험
//   A) BlurNaive  전역 메모리에서 스레드마다 25번 읽는다
//   B) BlurShared 타일 + halo 를 공유 메모리에 한 번 올리고 거기서 25번 읽는다
//
// 사용법:  05_BlurShared.exe [N]      (기본 N = 2048, 정사각 N x N)

#include "cu_common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define RADIUS  2                          // 5x5 박스 블러
#define TAPS    ((2 * RADIUS + 1) * (2 * RADIUS + 1))

#define TILE_X  32                         // blockDim.x = 32 → warp 가 가로 한 줄 (02, 04 의 교훈)
#define TILE_Y  8                          // 32 x 8 = 256 스레드, 스레드 하나가 출력 1픽셀
#define HALO_W  (TILE_X + 2 * RADIUS)      // 36
#define HALO_H  (TILE_Y + 2 * RADIUS)      // 12

// ---------------------------------------------------------------- 경계 처리
// 이미지 밖을 읽으면 가장자리 값을 복제한다(clamp). CPU 기준 구현도 똑같이 해야 한다.
__host__ __device__ inline int Clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float SampleClamped(const float* img, int x, int y, int w, int h) {
    return img[static_cast<size_t>(Clamp(y, 0, h - 1)) * w + Clamp(x, 0, w - 1)];
}

// ---------------------------------------------------------------- A) 순진한 블러
// 스레드마다 25번 전역 메모리를 읽는다.
// 이웃 스레드가 읽는 범위와 20칸이 겹치지만, 그걸 알 방법이 없어 각자 또 읽는다.
// (L1/L2 캐시가 어느 정도는 건져주지만, 명시적으로 아끼는 것만 못하다)
__global__ void BlurNaive(const float* in, float* out, int w, int h) {
    const int x = blockIdx.x * TILE_X + threadIdx.x;
    const int y = blockIdx.y * TILE_Y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    float sum = 0.0f;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            sum += SampleClamped(in, x + dx, y + dy, w, h);
        }
    }
    out[static_cast<size_t>(y) * w + x] = sum * (1.0f / TAPS);
}

// ---------------------------------------------------------------- B) 공유 메모리 + halo
// 1단계 블록이 필요한 영역 전체(타일 + 테두리 R칸)를 공유 메모리에 올린다
// 2단계 __syncthreads()  ← 모두가 다 올릴 때까지 기다린다. 없으면 남의 빈칸을 읽는다
// 3단계 공유 메모리에서 25번 읽어 평균을 낸다 (전역 메모리는 더 안 건드린다)
__global__ void BlurShared(const float* in, float* out, int w, int h) {
    __shared__ float tile[HALO_H][HALO_W];

    // 이 블록이 읽어야 할 영역의 좌상단 (halo 포함이라 R 만큼 왼쪽/위로 나간다)
    const int x0 = blockIdx.x * TILE_X - RADIUS;
    const int y0 = blockIdx.y * TILE_Y - RADIUS;

    // 읽을 칸(36x12=432)이 스레드 수(256)보다 많다 → 한 스레드가 두 칸씩 맡도록 돈다.
    // 블록 안에서 납작한 번호를 만들어 쓴다. (2D 좌표는 공유 메모리 인덱싱엔 불편하다)
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    for (int i = tid; i < HALO_W * HALO_H; i += nthreads) {
        const int lx = i % HALO_W;
        const int ly = i / HALO_W;
        tile[ly][lx] = SampleClamped(in, x0 + lx, y0 + ly, w, h);
    }

    __syncthreads();

    const int x = blockIdx.x * TILE_X + threadIdx.x;
    const int y = blockIdx.y * TILE_Y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    // 내 픽셀은 타일 안에서 (threadIdx + RADIUS) 위치다
    float sum = 0.0f;
    for (int dy = 0; dy <= 2 * RADIUS; ++dy) {
        for (int dx = 0; dx <= 2 * RADIUS; ++dx) {
            sum += tile[threadIdx.y + dy][threadIdx.x + dx];
        }
    }
    out[static_cast<size_t>(y) * w + x] = sum * (1.0f / TAPS);
}

// ---------------------------------------------------------------- CPU 기준
// 입력을 0~255 정수값 float 으로 만들었기 때문에, 25개를 더한 중간합도 정확히 표현된다
// (최대 6375 < 2^24). 마지막 나눗셈 한 번만 반올림이 생기는데 CPU/GPU 가 같은 연산을
// 같은 순서로 하므로 결과가 비트 단위로 일치해야 한다 → mismatch 는 0 이어야 정상.
static void BlurCpu(const std::vector<float>& in, std::vector<float>& out, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
                for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                    const int sy = Clamp(y + dy, 0, h - 1);
                    const int sx = Clamp(x + dx, 0, w - 1);
                    sum += in[static_cast<size_t>(sy) * w + sx];
                }
            }
            out[static_cast<size_t>(y) * w + x] = sum * (1.0f / TAPS);
        }
    }
}

static size_t CountMismatch(const std::vector<float>& ref, const std::vector<float>& got) {
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) {
            ++bad;
        }
    }
    return bad;
}

struct Result {
    double ms;
    size_t bad;
};

template <typename LaunchFn>
static Result RunAndCheck(LaunchFn launch, const float* dOut,
                          const std::vector<float>& expect,
                          std::vector<float>& scratch, int repeat) {
    const size_t bytes = expect.size() * sizeof(float);

    launch();                                   // 워밍업
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    std::vector<double> samples;
    samples.reserve(repeat);
    for (int r = 0; r < repeat; ++r) {
        t.Start();
        launch();
        CUDA_CHECK_LAUNCH();
        t.Stop();
        samples.push_back(t.ElapsedMs());
    }

    CUDA_CHECK(cudaMemcpy(scratch.data(), dOut, bytes, cudaMemcpyDeviceToHost));

    Result res;
    res.ms  = Median(samples);
    res.bad = CountMismatch(expect, scratch);
    return res;
}

int main(int argc, char** argv) {
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지

    const int n = (argc > 1) ? std::atoi(argv[1]) : 2048;
    if (n <= 0) {
        std::printf("N must be >= 1\n");
        return 1;
    }

    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);
    const int kRepeat = 15;

    // 주의: nvcc 프론트엔드는 /utf-8 을 못 받는다.
    //       문자열이 한글로 끝나면 바로 뒤 \n 이 깨진다 → 항상 ASCII 로 끝낼 것.
    std::printf("=== 05_BlurShared  %d x %d, %dx%d box blur (%d taps) ===\n\n",
                n, n, 2 * RADIUS + 1, 2 * RADIUS + 1, TAPS);

    // 스레드당 전역 읽기 횟수 — 이 예제의 핵심 숫자
    const double sharedLoadsPerPixel =
        static_cast<double>(HALO_W * HALO_H) / (TILE_X * TILE_Y);
    std::printf("타일 %dx%d + halo %d -> 공유 메모리 %dx%d = %d cells\n",
                TILE_X, TILE_Y, RADIUS, HALO_W, HALO_H, HALO_W * HALO_H);
    std::printf("출력 1픽셀당 전역 읽기: naive %d회  vs  shared %.2f회  (%.1fx)\n\n",
                TAPS, sharedLoadsPerPixel, TAPS / sharedLoadsPerPixel);

    // ---- 입력: 0~255 정수값 (이미지처럼). 정확히 표현돼서 비트 비교가 가능하다.
    std::vector<float> hIn(elems);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            hIn[static_cast<size_t>(y) * n + x] =
                static_cast<float>((x * 7 + y * 13) % 256);
        }
    }

    // ---- CPU 기준
    std::printf("CPU reference...");
    std::fflush(stdout);
    std::vector<float> hRef(elems);
    CpuTimer cpu;
    cpu.Start();
    BlurCpu(hIn, hRef, n, n);
    const double cpuMs = cpu.ElapsedMs();
    std::printf(" %.1f ms\n\n", cpuMs);

    // ---- 디바이스
    float* dIn  = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dIn,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE_X, TILE_Y);
    const dim3 grid(DivUp(n, TILE_X), DivUp(n, TILE_Y));

    std::vector<float> scratch(elems);
    auto clearOut = [&]() { CUDA_CHECK(cudaMemset(dOut, 0, bytes)); };

    clearOut();
    const Result rNaive = RunAndCheck(
        [&] { BlurNaive<<<grid, block>>>(dIn, dOut, n, n); }, dOut, hRef, scratch, kRepeat);

    clearOut();
    const Result rShared = RunAndCheck(
        [&] { BlurShared<<<grid, block>>>(dIn, dOut, n, n); }, dOut, hRef, scratch, kRepeat);

    // ---- 표
    std::printf("| 커널 | kernel ms | vs naive | vs CPU | mismatch |\n");
    std::printf("|---|---:|---:|---:|---:|\n");
    std::printf("| A) BlurNaive  | %8.3f | %6.2fx | %6.1fx | %zu |\n",
                rNaive.ms, 1.0, cpuMs / rNaive.ms, rNaive.bad);
    std::printf("| B) BlurShared | %8.3f | %6.2fx | %6.1fx | %zu |\n",
                rShared.ms, rNaive.ms / rShared.ms, cpuMs / rShared.ms, rShared.bad);

    std::printf("\n[참고] 공유 메모리 사용량 %zu bytes/block  (블록당 한도는 00_DeviceQuery 참고)\n",
                sizeof(float) * HALO_W * HALO_H);
    std::printf("       shared 가 naive 의 %.1f배 미만이면, L1/L2 캐시가 이미 상당 부분 건져주고 있다는 뜻이다.\n",
                TAPS / sharedLoadsPerPixel);

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    return 0;
}
