// 06_Sanitizer — S4 검증: 눈으로 못 잡는 버그를 도구로 잡는다
//
// 지금까지 쌓인 숙제를 여기서 갚는다.
//   01 README  "if (i < n) 를 빼면 무슨 일이 생기나? (S4 compute-sanitizer 로 다시 확인)"
//   03 main.cu "실행 중 에러(경계 밖 접근)는 sticky -> S4 compute-sanitizer"
//   04 README  "__syncthreads() 를 지우면 항상 틀리나, 가끔 틀리나?"
//   05         racy 커널에서 "실행마다 답이 다른" 현상을 직접 봤다
//
// 이 예제의 버그들은 전부 이런 성질을 가진다.
//   - 컴파일 경고가 없다
//   - 크래시도 안 난다 (대부분)
//   - 테스트를 통과하기도 한다 (가끔)
//   -> 눈과 단위 테스트로는 못 잡는다. 도구가 필요하다.
//
// compute-sanitizer 의 네 가지 도구
//   memcheck   (기본)  배열 밖 접근, 잘못된 free, 정렬 위반
//   racecheck         공유 메모리 경쟁 상태 (전역 메모리는 안 본다)
//   initcheck         초기화 안 된 전역 메모리 읽기
//   synccheck         잘못된 __syncthreads() 사용
//
// 사용법:  06_Sanitizer.exe [mode]
//   0  정상 동작 (기준)
//   1  배열 밖 접근        -> memcheck   + sticky 에러 증명
//   2  공유 메모리 경쟁      -> racecheck
//   3  초기화 안 된 메모리    -> initcheck
//
// 한 번에 하나씩만 돌린다. 1번은 CUDA 컨텍스트를 죽이기 때문에
// 같은 프로세스 안에서 다른 모드를 이어서 돌릴 수 없다 (그것 자체가 1번의 교훈이다).

#include "cu_common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------- 공통
// CUDA_CHECK 는 에러를 만나면 exit() 한다. 여기서는 에러가 난 뒤에도
// 계속 진행해서 "그 다음에 무슨 일이 생기는지" 를 봐야 하므로,
// 죽지 않고 찍기만 하는 버전을 따로 쓴다.
static void Report(const char* what, cudaError_t e) {
    std::printf("    %-34s %s\n", what, cudaGetErrorName(e));
}

// ---------------------------------------------------------------- 0) 정상
__global__ void SafeWrite(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {              // <- 이 한 줄이 있고 없고의 차이
        out[i] = i;
    }
}

// ---------------------------------------------------------------- 1) 배열 밖 접근
// 01 의 가드를 뺀 것. grid * block 은 n 보다 크게 반올림되므로
// 남는 쓰레드들이 배열 밖에 쓴다.
//   n = 1000, block = 256  ->  grid = 4  ->  쓰레드 1024개
//   1000 ~ 1023 번 쓰레드 24명이 할당 범위를 넘어선다
__global__ void OutOfBoundsWrite(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i;               // 가드 없음. i 가 n 이상이어도 그냥 쓴다
}

// 같은 "배열 밖 쓰기" 인데 훨씬 멀리 나간다. 할당된 페이지를 완전히 벗어나므로
// 이번에는 하드웨어가 잡아낸다 -> cudaErrorIllegalAddress (sticky).
// 위의 OutOfBoundsWrite 와 비교하면 "얼마나 멀리 나갔느냐" 가 에러 발생 여부를
// 가른다는 것을 알 수 있다. 즉 에러가 안 났다고 안전한 것이 아니다.
__global__ void FarOutOfBoundsWrite(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i + (1 << 26)] = i;   // 256 MB 너머. 매핑되지 않은 주소
}

// ---------------------------------------------------------------- 2) 공유 메모리 경쟁
// 04 에서 "__syncthreads() 를 지우면?" 하고 남겨둔 질문의 실물.
//
// 내가 쓴 칸이 아니라 "옆 쓰레드가 쓴 칸" 을 읽는다.
//   s[t] = t * 10;          <- 내가 쓴다
//   out[t] = s[(t+1) % B];  <- 옆 쓰레드가 쓴 것을 읽는다
//
// 같은 warp 안이면 명령이 같이 진행되므로 우연히 맞는다.
// warp 경계를 넘는 순간(t=31 이 s[32] 를 읽을 때) 순서가 보장되지 않는다.
// -> 틀릴 수도 있고 맞을 수도 있다. 그래서 더 위험하다.
__global__ void SharedRace(int* out, int n) {
    __shared__ int s[256];
    int t = threadIdx.x;

    s[t] = t * 10;

    // __syncthreads();      <- 여기 있어야 한다. 일부러 뺐다

    out[t] = s[(t + 1) % blockDim.x];
}

__global__ void SharedSafe(int* out, int n) {
    __shared__ int s[256];
    int t = threadIdx.x;

    s[t] = t * 10;
    __syncthreads();          // <- 있는 버전
    out[t] = s[(t + 1) % blockDim.x];
}

// ---------------------------------------------------------------- 3) 초기화 안 된 메모리
// cudaMalloc 은 0 으로 채워주지 않는다. malloc 과 같다.
// 05 에서 clearOutput() 으로 매번 cudaMemset 한 이유가 이것이다.
// 쓰레기값을 읽어도 "값이 있긴 하므로" 커널은 멀쩡히 돌고 답만 이상해진다.
__global__ void SumFrom(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, in[i]);
    }
}

// ---------------------------------------------------------------- 모드 0
static void RunSafe() {
    std::printf("[0] 정상 동작 (기준)\n\n");

    const int n = 1000;
    const int block = 256;
    const int grid = DivUp(n, block);

    int* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(int)));

    SafeWrite<<<grid, block>>>(d, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h(n, -1);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] != i) {
            ++bad;
        }
    }
    std::printf("    mismatch = %d\n", bad);
    std::printf("    sanitizer 를 걸어도 아무 보고가 없어야 정상이다.\n");

    CUDA_CHECK(cudaFree(d));
}

// ---------------------------------------------------------------- 모드 1
static void RunOutOfBounds() {
    std::printf("[1] 배열 밖 접근  ->  compute-sanitizer --tool memcheck\n\n");

    const int n = 1000;
    const int block = 256;
    const int grid = DivUp(n, block);

    std::printf("    n = %d, block = %d, grid = %d\n", n, block, grid);
    std::printf("    쓰레드 %d개 중 %d개가 배열 밖에 쓴다 (가드 없음)\n\n",
                grid * block, grid * block - n);

    int* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(int)));

    // 여기서부터는 CUDA_CHECK 를 안 쓴다. 에러가 난 뒤의 상태를 봐야 하기 때문이다.

    // ---- A) 살짝 넘침 (24칸)
    std::printf("    [A] 24칸만 넘어가 본다\n");
    OutOfBoundsWrite<<<grid, block>>>(d, n);
    Report("런치 직후 cudaGetLastError", cudaGetLastError());
    Report("cudaDeviceSynchronize", cudaDeviceSynchronize());

    std::printf("\n    아마 둘 다 cudaSuccess 일 것이다. 에러가 안 난다.\n");
    std::printf("    cudaMalloc 은 요청한 4000 바이트보다 넉넉하게 잡아주기 때문에,\n");
    std::printf("    조금 넘친 쓰기는 '할당된 페이지 안' 이라 하드웨어가 못 잡는다.\n");
    std::printf("    남의 데이터를 조용히 망가뜨리고 지나간다. 이것이 최악의 경우다.\n");
    std::printf("    -> memcheck 를 걸면 24건이 전부 보고된다. 도구만이 잡는다.\n\n");

    // ---- B) 멀리 넘침 -> 이번엔 하드웨어가 잡는다
    std::printf("    [B] 이번엔 256 MB 너머로 나가 본다\n");
    FarOutOfBoundsWrite<<<grid, block>>>(d);
    Report("런치 직후 cudaGetLastError", cudaGetLastError());
    Report("cudaDeviceSynchronize", cudaDeviceSynchronize());

    std::printf("\n    런치는 성공이고 동기화에서 에러가 났다.\n");
    std::printf("    실행 중 에러는 비동기라 동기화 지점에서야 드러난다 (03 에서 본 그것).\n\n");

    // ---- sticky 증명. 03 에서 "일부러 내지 않는다" 고 미뤄둔 부분이다.
    std::printf("    [sticky 확인] 이제 아무 상관없는 API 를 불러본다\n");
    int* d2 = nullptr;
    Report("cudaMalloc (새 할당)", cudaMalloc((void**)&d2, 16));
    Report("cudaGetLastError", cudaGetLastError());
    Report("cudaGetLastError (또)", cudaGetLastError());
    int* d3 = nullptr;
    Report("cudaMalloc (다시 한 번)", cudaMalloc((void**)&d3, 16));

    std::printf("\n    메모리 할당조차 실패한다. 커널과 아무 상관없는 호출인데도 그렇다.\n");
    std::printf("    주목할 것: cudaGetLastError 를 두 번 부르면 '마지막 에러' 슬롯은\n");
    std::printf("    비워져서 cudaSuccess 가 나온다. 그런데 바로 다음 cudaMalloc 은\n");
    std::printf("    또 실패한다. 슬롯만 비워졌을 뿐 컨텍스트는 여전히 죽어 있다.\n");
    std::printf("    이것이 sticky 다. 03 의 런치 설정 에러는 Get 한 번으로 완전히\n");
    std::printf("    치워졌고 이후 커널도 정상 동작했다. 거기와 비교해 볼 것.\n");
    std::printf("    복구 방법은 프로세스를 다시 띄우는 것뿐이다.\n\n");

    std::printf("    [A] 와 [B] 의 차이가 요점이다. 같은 '배열 밖 쓰기' 인데\n");
    std::printf("    얼마나 멀리 나갔느냐로 에러 발생 여부가 갈린다.\n");
    std::printf("    에러가 안 났다고 안전한 것이 아니다.\n");
}

// ---------------------------------------------------------------- 모드 2
static void RunRace() {
    std::printf("[2] 공유 메모리 경쟁  ->  compute-sanitizer --tool racecheck\n\n");

    const int block = 256;
    const int n = block;
    const int kRuns = 10;

    int* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(int)));
    std::vector<int> h(n);

    // 정답: out[t] = ((t+1) % block) * 10
    auto countBad = [&]() {
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int t = 0; t < n; ++t) {
            if (h[t] != ((t + 1) % block) * 10) {
                ++bad;
            }
        }
        return bad;
    };

    std::printf("    __syncthreads() 있는 버전을 %d번:\n      ", kRuns);
    for (int r = 0; r < kRuns; ++r) {
        CUDA_CHECK(cudaMemset(d, -1, n * sizeof(int)));
        SharedSafe<<<1, block>>>(d, n);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("%d ", countBad());
    }
    std::printf("  <- 전부 0 이어야 한다\n\n");

    std::printf("    __syncthreads() 없는 버전을 %d번:\n      ", kRuns);
    for (int r = 0; r < kRuns; ++r) {
        CUDA_CHECK(cudaMemset(d, -1, n * sizeof(int)));
        SharedRace<<<1, block>>>(d, n);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("%d ", countBad());
    }
    std::printf("  <- 틀린 개수\n\n");

    std::printf("    0 이 섞여 나올 수 있다. 운이 좋으면 맞는다는 뜻이다.\n");
    std::printf("    에러도 안 나고 테스트도 통과하는데 코드는 틀렸다.\n");
    // 주의: 한글 바로 뒤에 \ 를 쓰면 안 된다 (nvcc 가 /utf-8 을 못 받아 삼킨다).
    //       따옴표가 필요하면 작은따옴표를 쓰거나 ASCII 문자를 사이에 둔다.
    std::printf("    racecheck 는 결과가 맞든 틀리든 '위험한 접근' 자체를 보고한다.\n");

    CUDA_CHECK(cudaFree(d));
}

// ---------------------------------------------------------------- 모드 3
static void RunUninitialized() {
    std::printf("[3] 초기화 안 된 메모리  ->  compute-sanitizer --tool initcheck\n\n");

    const int n = 1024;
    const int block = 256;
    const int grid = DivUp(n, block);

    int* dIn = nullptr;
    int* dOut = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dIn, n * sizeof(int)));    // memset 하지 않는다
    CUDA_CHECK(cudaMalloc((void**)&dOut, sizeof(int)));
    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(int)));

    SumFrom<<<grid, block>>>(dIn, dOut, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    int sum = 0;
    CUDA_CHECK(cudaMemcpy(&sum, dOut, sizeof(int), cudaMemcpyDeviceToHost));

    std::printf("    cudaMalloc 만 하고 cudaMemset 을 안 한 배열의 합: %d\n", sum);
    std::printf("    0 이 나올 수도, 쓰레기값이 나올 수도 있다.\n");
    std::printf("    0 이 나왔다면 그건 우연이다 (직전에 그 자리를 쓴 사람이 없었을 뿐).\n\n");
    std::printf("    cudaMalloc 은 malloc 과 같아서 0 으로 채워주지 않는다.\n");
    std::printf("    05 에서 clearOutput() 으로 매번 cudaMemset 한 이유가 이것이다.\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

// ---------------------------------------------------------------- main
static void PrintUsage(const char* exe) {
    std::printf("사용법: %s [mode]\n\n", exe);
    std::printf("  0  정상 동작 (기준)\n");
    std::printf("  1  배열 밖 접근        -> memcheck   + sticky 에러 증명\n");
    std::printf("  2  공유 메모리 경쟁      -> racecheck\n");
    std::printf("  3  초기화 안 된 메모리    -> initcheck\n\n");
    std::printf("도구를 걸어 돌리는 법 (CUDA 설치 폴더의 compute-sanitizer):\n\n");
    std::printf("  compute-sanitizer --tool memcheck  %s 1\n", exe);
    std::printf("  compute-sanitizer --tool racecheck %s 2\n", exe);
    std::printf("  compute-sanitizer --tool initcheck %s 3\n\n", exe);
    std::printf("먼저 도구 없이 돌려서 '아무 일도 없어 보이는' 것을 확인하고,\n");
    std::printf("그 다음 도구를 걸어 무엇이 잡히는지 비교하는 것이 요점이다.\n");
}

int main(int argc, char** argv) {
    EnableUtf8Console();      // 콘솔 한글 깨짐 방지

    if (argc < 2) {
        PrintUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    const int mode = std::atoi(argv[1]);
    std::printf("=== 06_Sanitizer  mode %d ===\n\n", mode);

    switch (mode) {
    case 0:
        RunSafe();
        break;
    case 1:
        RunOutOfBounds();
        break;
    case 2:
        RunRace();
        break;
    case 3:
        RunUninitialized();
        break;
    default:
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::printf("\n");
    return EXIT_SUCCESS;
}
