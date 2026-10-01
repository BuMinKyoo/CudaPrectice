// 04_Transpose — S2 메모리: coalescing 과 공유 메모리
//
// 전치(transpose)는 계산이 없다. 값을 옮기기만 한다.
// 그래서 성능을 결정하는 건 오직 "메모리를 어떤 순서로 건드리는가" 하나뿐이다.
//
// 실험 — 네 커널이 옮기는 데이터 양은 완전히 같다. 방법만 다르다.
//   A) Copy                  전치가 아님. 읽기·쓰기 모두 연속 → 이 GPU 의 상한선
//   B) TransposeNaive        읽기는 연속인데 쓰기가 N 칸씩 건너뛴다 → 쓰기 비연속
//   C) TransposeShared       타일을 공유 메모리에 올려 읽기·쓰기 둘 다 연속으로
//   D) TransposeSharedPadded C + 패딩 한 칸으로 bank conflict 제거
//
// 사용법:  04_Transpose.exe [N]      (기본 N = 4096, 정사각 N x N)

#include "cu_common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// 블록 하나가 32x32 타일을 맡는다. 블록 자체는 32x8 = 256 스레드라,
// 스레드 하나가 세로로 4칸(32/8)을 처리한다.
//   32  : warp 크기. 한 warp 가 가로 한 줄을 통째로 읽게 만든다 (02 에서 본 그것)
//   256 : 02 의 블록 모양 스윕에서 제일 빨랐던 크기
#define TILE_DIM   32
#define BLOCK_ROWS 8

// ---------------------------------------------------------------- A) 기준선
// 전치를 안 하고 그냥 복사한다. 읽기도 쓰기도 연속이라 이 GPU 가 낼 수 있는
// 최고 속도에 가깝다. 아래 전치 커널들이 여기에 얼마나 근접하는지가 관전 포인트.
__global__ void Copy(const float* in, float* out, int n) {
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = in[(y + j) * n + x];
        }
    }
}

// ---------------------------------------------------------------- B) 순진한 전치
// out[x][y] = in[y][x] 를 그대로 옮겨 적었다. 논리적으로는 맞다(mismatch 0).
// 느린 이유는 주소에 있다. warp 안에서 threadIdx.x 가 0 -> 31 로 갈 때,
// n = 4096 / float 이면 한 행이 4096 * 4 = 16,384 바이트(16KB)다.
//
//   읽기  in[(y+j)*n + x]    x 가 1 늘면 주소 +4 바이트
//                            0, 4, 8, 12, ... 124      -> 128바이트 연속        OK
//
//   쓰기  out[x*n + (y+j)]   x 가 1 늘면 주소 +16,384 바이트 (한 행 통째)
//                            0, 16K, 32K, 48K, ...     -> 16KB 씩 흩어짐        BAD
//
// 왜 이게 문제인가 — GPU 는 VRAM 을 32바이트 단위로 주고받는다.
// 4바이트만 필요해도 32바이트를 통째로 싣고 온다. 그래서:
//
//                     트랜잭션 수   한 번에 쓰는 양   활용률
//   읽기 (연속 128B)       4 번        32 / 32 B      100%
//   쓰기 (흩어진 32곳)    32 번         4 / 32 B      12.5%   <- 8배 낭비
//
// 옮긴 데이터 양은 읽기와 똑같은데 트랜잭션만 8배다.
// 트럭 비유로는 "트럭 수(occupancy)" 가 아니라 "트럭에 얼마나 실었나" 의 문제다.
// 쓰기 한쪽만 망가졌는데 전체가 Copy 의 26% 로 떨어진다.
__global__ void TransposeNaive(const float* in, float* out, int n) {
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[x * n + (y + j)] = in[(y + j) * n + x];
        }
    }
}

// ---------------------------------------------------------------- C) 공유 메모리 타일
// 핵심 아이디어: 전역 메모리에서는 절대 흩어지지 말고, 뒤집는 일은 공유 메모리에서 하자.
//   1단계  전역 → 공유       연속으로 읽어서 타일에 담는다
//   2단계  __syncthreads()   블록 안 모든 스레드가 담기를 끝낼 때까지 기다린다
//   3단계  공유 → 전역       타일을 가로세로 바꿔 읽되, 쓰기는 연속이 되도록
//
// ─── __shared__ 란 ───
// SM 안에 있는 작고 빠른 메모리다. 흩어져 읽어도 전역 메모리보다 훨씬 싸다.
//                위치        지연          크기
//   레지스터     스레드 전용  ~1 사이클     스레드당 수십 개
//   __shared__   SM 안       ~30 사이클    블록당 최대 48KB   <- 여기
//   전역(VRAM)   칩 바깥     ~500 사이클   수 GB
//
// 헷갈리기 쉬운 점: 커널 함수 안에 선언하지만 지역 변수가 아니다.
//   int x = ...;                    스레드마다 하나씩  -> 256개 존재
//   __shared__ float tile[32][32];  블록당 하나       -> 256 스레드가 같은 것을 본다
// 한 스레드가 쓴 값을 다른 스레드가 읽을 수 있고, 그게 전치를 가능하게 한다.
//
// 수명은 블록과 같다. 블록이 끝나면 사라지고 다른 블록은 볼 수 없다.
// (블록이 SM 하나에 통째로 들어가야 하는 이유가 이것이다 — 공유 메모리가 그 SM 에 붙어 있다)
//
// occupancy 에도 영향을 준다. 여기서는 tile 이 블록당 4,224 바이트라
// 98,304 / 4,224 = 23 블록까지 가능해서 다른 제약(블록 32, 스레드 2048/256=8)이
// 먼저 걸리지만, 공유 메모리를 많이 쓰면 이것 때문에 블록이 덜 올라갈 수 있다.
__global__ void TransposeShared(const float* in, float* out, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];   // 연속 읽기
        }
    }

    // ─── __syncthreads() 란 ───
    // 블록 안 모든 스레드가 이 줄에 도착할 때까지, 먼저 온 스레드는 기다린다.
    //
    // 여기서 왜 반드시 필요한가 — 쓴 자리와 읽는 자리가 다르기 때문이다.
    // 스레드 (tx=5, ty=2) 를 따라가 보면:
    //   1단계  tile[ty+j][tx] = tile[2][5]   <- 내가 쓴다
    //   3단계  tile[tx][ty+j] = tile[5][2]   <- 내가 읽는다
    // tile[5][2] 를 쓴 것은 ty+j=5, tx=2 인 스레드, 즉 (tx=2, ty=5) 다. 남이다.
    //
    // 게다가 둘은 다른 warp 다 (tid = ty*32 + tx):
    //   스레드 (5,2) -> tid  69 -> warp 2
    //   스레드 (2,5) -> tid 162 -> warp 5
    // warp 는 서로 독립이라 warp 2 가 warp 5 보다 먼저 달릴 수 있다.
    // sync 가 없으면 warp 2 는 아직 아무도 안 쓴 칸을 읽는다
    //   -> 그것도 "항상" 이 아니라 스케줄링 운에 따라 "가끔" 틀린다. 최악의 버그다.
    //
    // 규칙 둘:
    //   1. 블록 안에서만 동작한다 (다른 블록을 기다리는 수단은 없다)
    //   2. 모든 스레드가 도달해야 한다 — 조건문 안에 넣으면 안 된다
    //      if (threadIdx.x < 16) { __syncthreads(); }  <- 절반만 도착. 미정의 동작
    __syncthreads();

    // 출력에서 내 블록이 맡을 자리는 입력과 가로세로가 뒤바뀐 위치다
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];  // 연속 쓰기
        }
    }

    // 이제 전역 메모리는 읽기도 쓰기도 연속이다. 그런데 Copy 의 60% 밖에 안 나온다.
    // 범인은 tile[threadIdx.x][...] — 타일을 "세로로" 읽는 꼴이기 때문이다.
    //
    // 공유 메모리는 32개 bank 로 나뉘어 있고  bank 번호 = (주소 / 4) % 32 다.
    //   32개 스레드가 서로 다른 bank  → 동시에 처리
    //   여러 스레드가 같은 bank        → 줄 서서 순서대로
    //
    // warp 안에서 tx(threadIdx.x) 가 0 -> 31 로 갈 때, 한 행이 32 float 이면:
    //
    //   tile[tx][c] 위치 = tx * 32 + c
    //   bank = (tx * 32 + c) % 32 = c % 32        <- tx 가 사라진다!
    //
    //   tx   :  0    1    2   ...  31
    //   bank :  c    c    c   ...   c             <- 전부 같은 bank
    //
    //   → 32-way bank conflict. 한 번에 못 읽고 32번에 나눠 읽는다.  (해결은 D)
}

// ---------------------------------------------------------------- D) 패딩 한 칸
// C 와 코드가 딱 한 글자 다르다. 행 길이를 32 -> 33 으로 늘릴 뿐,
// 인덱스 계산식은 아래에서 한 글자도 안 바뀐다.
//
//   【C】 한 행이 32 float
//     tile[tx][c] 위치 = tx * 32 + c
//     bank = (tx * 32 + c) % 32 = c % 32              <- tx 가 사라진다
//     tx   :  0    1    2   ...  31
//     bank :  c    c    c   ...   c                   <- 전부 충돌        BAD
//
//   【D】 한 행이 33 float   (33 % 32 = 1 이 핵심)
//     tile[tx][c] 위치 = tx * 33 + c
//     bank = (tx * 33 + c) % 32 = (tx + c) % 32       <- tx 가 살아남는다
//     tx   :  0    1    2   ...  31
//     bank :  c   c+1  c+2  ... c+31  (mod 32)        <- 전부 다른 bank   OK
//
// 행을 하나 넘어갈 때마다 bank 가 1칸씩 밀리는 것이 전부다.
// 비용은 공유 메모리 32 * 1 * 4 = 128 바이트/블록 추가뿐이다.
__global__ void TransposeSharedPadded(const float* in, float* out, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];   // ← 이 +1 이 전부다

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ---------------------------------------------------------------- 헬퍼
// 전치는 값을 옮기기만 하므로 CPU 와 GPU 결과가 비트 단위로 같아야 한다.
// 계산이 없어서 부동소수점 오차가 끼어들 여지가 아예 없다
//   → mismatch 가 0 이 아니면 100% 인덱싱 버그다.
static void TransposeCpu(const std::vector<float>& in, std::vector<float>& out, int n) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            out[static_cast<size_t>(x) * n + y] = in[static_cast<size_t>(y) * n + x];
        }
    }
}

static size_t CountMismatch(const std::vector<float>& ref, const std::vector<float>& got) {
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != got[i]) ++bad;
    }
    return bad;
}

// 읽기 + 쓰기 = 원소 수 x 4바이트 x 2
static double BandwidthGBs(size_t elems, double ms) {
    const double bytes = 2.0 * static_cast<double>(elems) * sizeof(float);
    return bytes / (ms * 1.0e6);
}

struct Result {
    double ms;
    size_t bad;
};

// 워밍업 1회 + kRepeat 회 측정(중앙값), 그리고 CPU 기준과 비교.
template <typename LaunchFn>
static Result RunAndCheck(LaunchFn launch, const float* dOut,
                          const std::vector<float>& expect,
                          std::vector<float>& scratch, int repeat) {
    const size_t bytes = expect.size() * sizeof(float);

    launch();                                   // 워밍업 (첫 런치엔 컨텍스트/JIT 비용이 섞인다)
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

    const int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (n <= 0) {
        std::printf("N 은 1 이상이어야 합니다.\n");
        return 1;
    }

    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);
    const int kRepeat = 15;

    std::printf("=== 04_Transpose  %d x %d  (행렬 하나 %.1f MB) ===\n\n",
                n, n, bytes / 1048576.0);
    // 주의: nvcc 프론트엔드는 /utf-8 을 못 받는다. 문자열 끝을 한글로 두면
    //       바로 뒤 \n 이 깨져서 literal 로 찍힌다 → 항상 ASCII 로 끝낼 것.
    std::printf("타일 %dx%d, 블록 %dx%d = %d threads, 스레드당 %d\n\n",
                TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS,
                TILE_DIM * BLOCK_ROWS, TILE_DIM / BLOCK_ROWS);

    // ---- 호스트 데이터. 위치마다 다른 값이어야 전치 버그가 드러난다.
    std::vector<float> hIn(elems);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            hIn[static_cast<size_t>(y) * n + x] =
                static_cast<float>(y % 1000) * 1000.0f + static_cast<float>(x % 1000);
        }
    }

    // ---- CPU 기준 전치 (정답지). 쓰기가 캐시에 불친절해서 시간이 좀 걸린다.
    std::printf("CPU 기준 전치 계산 중...");
    std::fflush(stdout);
    std::vector<float> hRefT(elems);
    CpuTimer cpu;
    cpu.Start();
    TransposeCpu(hIn, hRefT, n);
    const double cpuMs = cpu.ElapsedMs();
    std::printf(" %.1f ms\n\n", cpuMs);

    // ---- 디바이스 메모리
    float* dIn  = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dIn,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE_DIM, BLOCK_ROWS);
    const dim3 grid(DivUp(n, TILE_DIM), DivUp(n, TILE_DIM));

    std::vector<float> scratch(elems);

    // 커널마다 dOut 을 지운다. 안 쓴 칸이 남아 우연히 통과하는 일을 막기 위해서다.
    auto clearOut = [&]() { CUDA_CHECK(cudaMemset(dOut, 0, bytes)); };

    clearOut();
    const Result rCopy = RunAndCheck(
        [&] { Copy<<<grid, block>>>(dIn, dOut, n); }, dOut, hIn, scratch, kRepeat);

    clearOut();
    const Result rNaive = RunAndCheck(
        [&] { TransposeNaive<<<grid, block>>>(dIn, dOut, n); }, dOut, hRefT, scratch, kRepeat);

    clearOut();
    const Result rShared = RunAndCheck(
        [&] { TransposeShared<<<grid, block>>>(dIn, dOut, n); }, dOut, hRefT, scratch, kRepeat);

    clearOut();
    const Result rPadded = RunAndCheck(
        [&] { TransposeSharedPadded<<<grid, block>>>(dIn, dOut, n); }, dOut, hRefT, scratch, kRepeat);

    // ---- 표
    std::printf("| 커널 | kernel ms | 실효 대역폭 | Copy 대비 | mismatch |\n");
    std::printf("|---|---:|---:|---:|---:|\n");

    struct Row { const char* name; const Result* r; };
    const Row rows[] = {
        {"A) Copy (기준선)",       &rCopy},
        {"B) TransposeNaive",      &rNaive},
        {"C) TransposeShared",     &rShared},
        {"D) TransposeShared+pad", &rPadded},
    };

    for (const Row& row : rows) {
        std::printf("| %-22s | %8.3f | %7.1f GB/s | %5.1f%% | %zu |\n",
                    row.name, row.r->ms, BandwidthGBs(elems, row.r->ms),
                    100.0 * rCopy.ms / row.r->ms, row.r->bad);
    }

    std::printf("\n[참고] CPU 전치 %.1f ms  ->  D 대비 %.0fx\n", cpuMs, cpuMs / rPadded.ms);
    std::printf("       A) Copy 는 전치를 하지 않으므로 비교 기준이 원본(hIn)이다.\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    return 0;
}
