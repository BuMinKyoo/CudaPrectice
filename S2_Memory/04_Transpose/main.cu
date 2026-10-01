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
// 그리고 덤으로 — A~D 는 원소를 "한 번씩" 읽는다. 공유 메모리가 접근 순서를 바꾸는
// 환승역 역할만 했다. E/F 는 원소를 "여러 번" 읽는 경우다. 같은 값을 반복해서
// 읽을 때 공유 메모리로 중복을 없애는 것 — 이게 공유 메모리의 본래 목적이다.
//   E) StencilNaive          가로 (2R+1)칸 합. 스레드마다 전역 메모리를 (2R+1)번 읽는다
//   F) StencilShared         필요한 (32+2R)칸을 공유 메모리에 한 번 올리고 거기서 읽는다
//   -> RADIUS 를 키우면 재사용이 커진다. 측정값은 README 참고 (R=2 는 차이가 0 이다)
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

// E/F 전용 — 가로 5칸(= 2*RADIUS+1) 합. 재사용 배수를 바꿔보려면 여기만 고친다.
#define RADIUS   2
#define TAPS     (2 * RADIUS + 1)
#define ROW_W    (TILE_DIM + 2 * RADIUS)   // 공유 메모리 행 길이 = 36

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

// ================================================================ 재사용 (E/F)
// A~D 는 원소를 한 번씩만 읽었다. 공유 메모리는 "순서"를 바꾸는 환승역이었다.
// E/F 는 같은 원소를 여러 번 읽는다. 공유 메모리로 "중복"을 없앤다.
//
// 문제: 가로 5칸 합   out[y][x] = in[y][x-2] + in[y][x-1] + in[y][x] + in[y][x+1] + in[y][x+2]
//   출력 1픽셀에 입력 5개가 필요한데, 바로 옆 픽셀도 그 중 4개를 똑같이 쓴다.
//
// 블록당 전역 읽기 (블록 = 32x8 = 256 스레드, 출력 256픽셀, R=2 일 때)
//   E) 256 픽셀 x 5 = 1280 회
//   F) 36 x 8       =  288 회     ->  4.44배 감소
//
// 그런데 R=2 에서는 실제 시간이 "전혀" 안 줄어든다 (측정 1.00x).
// L1 캐시가 중복 읽기를 이미 다 건져주고 있기 때문이다. RADIUS 를 키워서
// 재사용을 늘리면 캐시만으로 부족해지고 비로소 공유 메모리가 이긴다.
//   R= 2 (5칸)   트래픽 4.44x 감소  ->  시간 1.00x
//   R= 8 (17칸)  트래픽 11.3x 감소  ->  시간 1.21x
//   R=16 (33칸)  트래픽 16.5x 감소  ->  시간 1.62x
//   R=32 (65칸)  트래픽 21.7x 감소  ->  시간 1.87x
// "트래픽을 줄인 배수" 와 "시간이 줄어든 배수" 의 격차가 곧 캐시의 몫이다.
//
// halo(apron): 타일 가장자리를 계산하려면 타일 "밖" 이웃도 필요하다.
//   32칸을 출력하려면 왼쪽 2칸 + 오른쪽 2칸을 더 읽어야 한다 -> 36칸
//
// 여기는 가로 방향만 하므로 halo 가 좌우 2칸씩뿐이다. 그래서 읽을 칸(36x8=288)이
// 스레드 수(256)와 비슷해서 "전원 1칸 + 왼쪽 2명이 좌우 halo" 로 if 한 줄에 끝난다.
//
// [2D 로 확장할 때] 5x5 처럼 세로 halo 까지 필요하면 읽을 칸이 (32+4)x(8+4)=432 로
// 스레드 수(256)를 넘는다. 그러면 1:1 배정이 불가능해서 로딩 루프를 써야 한다.
// 2D 좌표로는 균등 배분이 불편하니 블록 안 번호를 납작하게 만들어 돌린다.
//
//   const int tid      = threadIdx.y * blockDim.x + threadIdx.x;   // 0~255
//   const int nthreads = blockDim.x * blockDim.y;
//   for (int i = tid; i < HALO_W * HALO_H; i += nthreads) {        // 블록 안 grid-stride
//       const int lx = i % HALO_W;     // 이 순서여야 warp 가 연속으로 읽는다
//       const int ly = i / HALO_W;     // (뒤집으면 세로로 읽어서 coalescing 깨짐)
//       tile[ly][lx] = SampleClamped(in, x0 + lx, y0 + ly, n);
//   }
//
// 2D 컨볼루션, 행렬곱 타일링 등 "타일이 스레드보다 큰" 경우에 계속 쓰는 패턴이다.
//
// 나눗셈(/5)을 일부러 안 한다. 합까지만 하면 입력이 정수값이라 중간합도 float 에
// 정확히 들어가서(최대 5x999999 < 2^24) CPU 와 비트 단위로 같아야 한다.
// -> A~D 와 똑같이 mismatch 0 을 검증 기준으로 쓸 수 있다.

// 이미지 밖은 가장자리 값을 복제한다(clamp). CPU 기준 구현도 똑같이 해야 한다.
__host__ __device__ inline int ClampIdx(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

__device__ inline float SampleClamped(const float* img, int x, int y, int n) {
    return img[static_cast<size_t>(ClampIdx(y, 0, n - 1)) * n + ClampIdx(x, 0, n - 1)];
}

// ---------------------------------------------------------------- E) 순진한 5칸 합
// 스레드마다 전역 메모리를 5번 읽는다. 이웃 스레드와 4칸이 겹치지만 알 방법이 없어 각자 또 읽는다.
// 주의 — 이 읽기는 coalescing 이 잘 된다. warp 32명이 dx 마다 연속된 32칸을 읽는다.
//        즉 B 커널과 달리 "패턴" 문제가 아니라 순전히 "횟수" 문제다.
__global__ void StencilNaive(const float* in, float* out, int n) {
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    if (x >= n || y >= n) {
        return;
    }

    float sum = 0.0f;
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
        sum += SampleClamped(in, x + dx, y, n);
    }
    out[static_cast<size_t>(y) * n + x] = sum;
}

// ---------------------------------------------------------------- F) 공유 메모리 + halo
// 1단계 블록이 필요한 36칸을 공유 메모리에 올린다 (전원 1칸 + tx 0,1 이 좌우 halo)
// 2단계 __syncthreads()   내 5칸 중 halo 는 "남이" 올린 칸이다
// 3단계 공유 메모리에서 5번 읽는다. 전역 메모리는 더 안 건드린다
__global__ void StencilShared(const float* in, float* out, int n) {
    __shared__ float row[BLOCK_ROWS][ROW_W];   // 8 x 36 = 1152 bytes

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * TILE_DIM + tx;
    const int y  = blockIdx.y * BLOCK_ROWS + ty;

    // 가운데 32칸은 전원이 하나씩 (row[ty][2] ~ row[ty][33])
    row[ty][tx + RADIUS] = SampleClamped(in, x, y, n);

    // 좌우 halo 4칸은 tx 0,1 두 명이 각각 2칸씩 맡는다
    //   row[ty][0], row[ty][1]    <- 내 블록 왼쪽 밖 2칸
    //   row[ty][34], row[ty][35]  <- 내 블록 오른쪽 밖 2칸
    if (tx < RADIUS) {
        row[ty][tx]                     = SampleClamped(in, x - RADIUS, y, n);
        row[ty][tx + RADIUS + TILE_DIM] = SampleClamped(in, x + TILE_DIM, y, n);
    }

    // 장벽 "앞에서" return 하면 안 된다. 일부 스레드만 빠져나가면 나머지는 영원히
    // 기다린다(deadlock). 그래서 경계 검사를 장벽 뒤로 미뤘다.
    // 위 로딩은 SampleClamped 가 범위를 잡아주므로 경계 블록에서도 안전하다.
    __syncthreads();

    if (x >= n || y >= n) {
        return;
    }

    // 내 픽셀은 row[ty][tx + RADIUS] 다. 루프를 -RADIUS..+RADIUS 대신 0..2*RADIUS 로
    // 돌려서 그 오프셋을 흡수했다 -> 식에 + RADIUS 가 안 보이는 이유.
    float sum = 0.0f;
    for (int dx = 0; dx <= 2 * RADIUS; ++dx) {
        sum += row[ty][tx + dx];
    }
    out[static_cast<size_t>(y) * n + x] = sum;
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

// E/F 의 정답지. 커널과 같은 clamp 규칙을 써야 한다.
static void StencilCpu(const std::vector<float>& in, std::vector<float>& out, int n) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float sum = 0.0f;
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                sum += in[static_cast<size_t>(y) * n + ClampIdx(x + dx, 0, n - 1)];
            }
            out[static_cast<size_t>(y) * n + x] = sum;
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
    std::printf(" %.1f ms\n", cpuMs);

    // ---- E/F 용 CPU 기준 (가로 5칸 합)
    std::printf("CPU 기준 5-tap 합 계산 중...");
    std::fflush(stdout);
    std::vector<float> hRefS(elems);
    CpuTimer cpuS;
    cpuS.Start();
    StencilCpu(hIn, hRefS, n);
    const double cpuStencilMs = cpuS.ElapsedMs();
    std::printf(" %.1f ms\n\n", cpuStencilMs);

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

    // E/F 는 스레드 하나가 출력 1픽셀이라 그리드 모양이 다르다.
    // (A~D 는 32x32 타일을 32x8 스레드가 4번 돌아 맡았다)
    const dim3 gridStencil(DivUp(n, TILE_DIM), DivUp(n, BLOCK_ROWS));

    clearOut();
    const Result rStNaive = RunAndCheck(
        [&] { StencilNaive<<<gridStencil, block>>>(dIn, dOut, n); }, dOut, hRefS, scratch, kRepeat);

    clearOut();
    const Result rStShared = RunAndCheck(
        [&] { StencilShared<<<gridStencil, block>>>(dIn, dOut, n); }, dOut, hRefS, scratch, kRepeat);

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

    // ---- E/F 표: 재사용 (A~D 와 성격이 달라 표를 따로 둔다)
    // A~D 는 읽기·쓰기 양이 모두 같아서 "실효 대역폭" 으로 줄세울 수 있었다.
    // E/F 는 읽는 양 자체가 다르다(5배). 그래서 대역폭 대신 "픽셀당 전역 읽기" 를 본다.
    const double readsNaive  = TAPS;
    const double readsShared = static_cast<double>(ROW_W * BLOCK_ROWS) /
                               (TILE_DIM * BLOCK_ROWS);
    std::printf("\n=== E/F) 재사용 — 가로 %d칸 합 ===\n", TAPS);
    std::printf("공유 메모리 %zu bytes/block, 출력 1픽셀당 전역 읽기 %.0f -> %.3f (%.2fx)\n\n",
                sizeof(float) * ROW_W * BLOCK_ROWS, readsNaive, readsShared,
                readsNaive / readsShared);

    std::printf("| 커널 | kernel ms | naive 대비 | 픽셀당 전역 읽기 | mismatch |\n");
    std::printf("|---|---:|---:|---:|---:|\n");
    std::printf("| E) StencilNaive  | %8.3f | %6.2fx | %6.0f | %zu |\n",
                rStNaive.ms, 1.0, readsNaive, rStNaive.bad);
    std::printf("| F) StencilShared | %8.3f | %6.2fx | %6.3f | %zu |\n",
                rStShared.ms, rStNaive.ms / rStShared.ms, readsShared, rStShared.bad);

    std::printf("\n[참고] CPU 5-tap %.1f ms  ->  F 대비 %.0fx\n",
                cpuStencilMs, cpuStencilMs / rStShared.ms);
    std::printf("       E->F 가 %.2fx 보다 작으면 L1/L2 캐시가 이미 중복 읽기를 건져준 것이다.\n",
                readsNaive / readsShared);

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    return 0;
}
