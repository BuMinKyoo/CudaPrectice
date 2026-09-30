// 06_Reduction — S3 리덕션: N개를 1개로 접기
//
// 04/05 는 "N개 입력 -> N개 출력" 이었다. 스레드마다 자기 칸이 따로 있어서
// 서로 간섭할 일이 없었다. 리덕션은 다르다. 출력이 하나다.
// 여러 스레드가 같은 결과에 기여해야 하니 처음으로 "협력"이 필요해진다.
//   - 누가 누구 값을 더할지 (공유 메모리 + __syncthreads)
//   - warp 안에서 갈라지면 어떻게 되는지 (divergence)
//   - 블록끼리는 어떻게 합칠지 (다중 패스 vs atomicAdd)
//
// 실험 — 여섯 커널이 구하는 값은 전부 똑같은 합계다. 접는 방법만 다르다.
//   A) ReduceDivergent      교과서 트리. tid % (2*s) 조건 -> warp divergence
//   B) ReduceNoDivergent    인덱스를 바꿔 divergence 제거. 대신 bank conflict
//   C) ReduceSequential     접는 방향을 뒤집어 divergence + conflict 둘 다 제거
//   D) ReduceGridStride     스레드당 여러 칸을 먼저 더한다 -> 트리 높이를 낮춘다
//   E) ReduceShuffle        마지막 warp 는 __shfl_down_sync 로, 공유 메모리 없이
//   F) ReduceShuffleAtomic  블록 결과를 atomicAdd -> 다중 패스를 1패스로
//
// 그리고 이 예제에는 04 에 없던 함정이 하나 더 있다.
// 04 는 "mismatch == 0" 이 정답이었다. 값을 옮기기만 했으니 비트가 같아야 했다.
// 리덕션은 더한다. float 덧셈은 결합법칙이 성립하지 않는다.
//   (a+b)+c != a+(b+c)
// 즉 더하는 순서가 바뀌면 답이 바뀐다. 아래 표의 상대오차를 보면
// 오히려 GPU 트리합이 CPU 순차합보다 정확하다. 이유는 README 에.
//
// 사용법:  06_Reduction.exe [N]      (기본 N = 1 << 24, 약 1678만개 / 64MB)

#include "cu_common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 블록 256 스레드. 2의 거듭제곱이어야 한다 — 트리로 반씩 접기 때문에
// 홀수가 끼면 짝이 없는 칸이 생긴다. (가정을 코드에 박아두는 것도 최적화다.
// 아래 커널들은 blockDim 이 2의 거듭제곱이라고 전제하고 쓴다)
#define BLOCK_SIZE 256

// D/E/F 에서 쓸 블록 수 상한과, 스레드 하나가 맡을 칸 수.
// 블록을 무한정 늘리는 대신 스레드당 일을 늘린다 -> 트리 패스가 줄어든다.
//   N=1678만 -> 1패스에 1024블록(출력 1024개) -> 2패스는 1블록으로 끝 -> 총 2패스
#define GRID_STRIDE_CAP   1024
#define ELEMS_PER_THREAD  4

// ---------------------------------------------------------------- A) 교과서 트리
// 가장 먼저 떠오르는 방법. 1칸 옆, 2칸 옆, 4칸 옆... 을 흡수해 나간다.
//
//   s=1:  t0+=t1   t2+=t3   t4+=t5   t6+=t7     (짝수 tid 만 일한다)
//   s=2:  t0+=t2            t4+=t6
//   s=4:  t0+=t4
//
// 문제는 `tid % (2*s) == 0` 이다. warp 32개 스레드가 같은 명령을 함께 실행하는데,
// 조건이 스레드마다 갈리면 GPU 는 참인 쪽과 거짓인 쪽을 "순차로" 실행한다 (divergence).
// s=1 일 때 warp 안 32개 중 16개만 일하고 나머지 16개는 놀면서 기다린다.
// s=2 면 8개, s=4 면 4개... 일하는 비율이 계속 절반으로 떨어진다.
//
// 게다가 % (나머지 연산)는 GPU 에서 비싸다. 정수 나눗셈 유닛이 따로 없다.
__global__ void ReduceDivergent(const float* in, float* out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;

    // 블록이 입력 끝을 넘어갈 수 있다. 남는 칸은 0 으로 채운다.
    // 합계의 항등원이 0 이라 결과에 영향이 없다
    // (최댓값 리덕션이면 0 이 아니라 -INF 를 넣어야 한다).
    sdata[tid] = (i < static_cast<unsigned>(n)) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        // __syncthreads() 는 if 밖에 있어야 한다.
        // 블록 안 "모든" 스레드가 도달해야 통과하는 장벽이라,
        // 일부만 들어가는 if 안에 넣으면 나머지는 영원히 기다린다 (deadlock).
        // 위 if 가 sdata 쓰기만 감싸고 장벽은 밖에 둔 이유가 이것이다.
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------- B) divergence 제거
// 하는 일은 A 와 똑같다. "일할 스레드를 앞으로 모았다"는 것만 다르다.
//   A: tid 0,2,4,6,...  가 일한다  -> warp 안에서 절반이 놀며 갈라진다
//   B: tid 0,1,2,3,...  가 일한다  -> 앞쪽 warp 는 전원 근무, 뒤쪽 warp 는 전원 퇴근
// warp 단위로 갈리면 divergence 가 아니다. 노는 warp 는 스케줄에서 아예 빠진다.
//
// 대신 새 문제가 생긴다. index = 2*s*tid 는 s 가 커질수록 성큼성큼 뛴다.
// 공유 메모리는 32개 bank 로 나뉘고 bank = (주소/4) % 32 인데 (04 의 그 bank),
// s=1 이면 스레드들이 0,2,4,... 를 건드려 2칸씩 뛴다 -> 2-way conflict.
// s=2 면 4칸씩 -> 4-way. s=16 이면 32칸씩 -> 32개 스레드가 전부 같은 bank.
__global__ void ReduceNoDivergent(const float* in, float* out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < static_cast<unsigned>(n)) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2)
    {
        const unsigned index = 2 * s * tid;
        if (index + s < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------- C) 접는 방향 뒤집기
// A/B 는 "옆칸"을 흡수했다. C 는 "반대쪽 절반"을 흡수한다.
//
//   s=128: t0 += t128, t1 += t129, ... t127 += t255
//   s=64 : t0 += t64,  t1 += t65,  ... t63  += t127
//   s=32 : ...
//
// 이 한 줄 차이로 두 문제가 동시에 사라진다.
//   divergence : 일하는 스레드가 항상 tid < s 라 앞쪽에 연속으로 모인다 (B 와 동일)
//   conflict   : 스레드 tid 가 건드리는 칸이 tid, tid+s 로 "연속"이다
//                -> 32개 스레드가 32개 다른 bank. 충돌 0
// 리덕션 코드에서 관용적으로 쓰는 형태가 바로 이것이다.
__global__ void ReduceSequential(const float* in, float* out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < static_cast<unsigned>(n)) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------- D) 스레드당 여러 칸
// A~C 는 스레드 하나가 값 하나를 맡았다. 그러면 스레드 수 = 입력 수 라서
// N = 1678만이면 블록이 65536개 생기고, 그 65536개를 또 접어야 한다 (패스 3번).
// 게다가 각 스레드는 딱 한 번 더하고는 로그 단계 내내 대부분 놀고 있다.
//
// 그래서 블록 수를 GRID_STRIDE_CAP 으로 묶어놓고, 남는 일은 grid-stride 루프로
// 스레드가 직접 더 가져간다 (01 에서 본 그 루프). 트리로 들어가기 전에
// 이미 대부분의 덧셈이 끝나 있고, 트리는 마무리만 한다.
//   - 읽기는 여전히 연속이다: 한 warp 가 stride 안에서 가로로 붙어 읽는다
//   - 패스가 3번(65536 -> 256 -> 1) 에서 2번(1024 -> 1) 으로 줄어든다
__global__ void ReduceGridStride(const float* in, float* out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const unsigned tid    = threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    // 레지스터에서 먼저 최대한 접는다. 공유 메모리보다 레지스터가 더 빠르다.
    float sum = 0.0f;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<unsigned>(n); i += stride)
    {
        sum += in[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------- warp 안에서 접기
// s 가 32 이하로 내려가면 일하는 스레드가 warp 하나 안에 다 들어간다.
// 그 시점부터 __syncthreads() 는 낭비다 — warp 는 원래 명령을 함께 실행하니까.
// 게다가 공유 메모리를 경유할 필요도 없다. 같은 warp 라면 레지스터를
// 직접 주고받을 수 있다.
//
//   __shfl_down_sync(mask, v, off) = "off 칸 위 레인의 v 값을 가져온다"
//   mask 는 참여할 레인 비트맵. 0xffffffff = 32개 전원.
//   이름에 _sync 가 붙은 이유: Volta 이후로 warp 안에서도 레인이 따로 놀 수 있어서
//   (independent thread scheduling), 누가 참여하는지 명시하도록 바뀌었다.
//   옛 코드의 `volatile float* s` 트릭은 이제 정답이 아니다.
__inline__ __device__ float WarpReduceSum(float v)
{
    for (int off = 16; off > 0; off >>= 1)
    {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// ---------------------------------------------------------------- E) 마지막 warp 는 shuffle
// D 와 같지만 트리를 s > 32 까지만 돌리고, 나머지는 warp 하나가 처리한다.
//   - __syncthreads() 호출이 8번 -> 3번으로
//   - 마지막 5단계의 공유 메모리 왕복이 사라진다
__global__ void ReduceShuffle(const float* in, float* out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const unsigned tid    = threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<unsigned>(n); i += stride)
    {
        sum += in[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // warp 하나에 들어갈 때까지만 공유 메모리로 접는다
    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 여기서 tid < 32 는 "warp 0 전원" 이다. 갈라지는 게 아니라
    // warp 단위로 잘리는 것이라 mask 0xffffffff 를 그대로 써도 된다.
    if (tid < 32)
    {
        float v = sdata[tid] + sdata[tid + 32];
        v = WarpReduceSum(v);
        if (tid == 0)
        {
            out[blockIdx.x] = v;
        }
    }
}

// ---------------------------------------------------------------- F) 1패스 + atomic
// E 까지는 블록 결과가 gridDim.x 개 남아서 커널을 한 번 더 돌려야 했다.
// 블록마다 대표 스레드가 atomicAdd 로 같은 칸에 더해버리면 패스가 1번으로 끝난다.
//   atomicAdd(p, v) = 다른 스레드와 겹치지 않게 *p += v 를 한 번에
// 블록당 딱 한 번만 부르는 게 요점이다. 스레드마다 부르면 경합이 256배가 된다.
//
// 대가가 두 개 있다.
//   1) 출력 칸을 미리 0 으로 지워야 한다 (아래 RunF 의 memset)
//   2) 더해지는 순서가 실행마다 달라진다 -> float 이라 결과도 실행마다 미세하게
//      달라진다. 재현성이 필요한 계산이라면 이건 못 쓴다.
__global__ void ReduceShuffleAtomic(const float* in, float* out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    const unsigned tid    = threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<unsigned>(n); i += stride)
    {
        sum += in[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        float v = sdata[tid] + sdata[tid + 32];
        v = WarpReduceSum(v);
        if (tid == 0)
        {
            atomicAdd(out, v);
        }
    }
}

// ---------------------------------------------------------------- 런치 래퍼
// 커널마다 블록 수 계산이 다르므로 함수로 묶는다.
// 반환값 = 이 패스가 만든 출력 개수. 1 이면 끝, 아니면 한 번 더 접어야 한다.
using RunFn = int (*)(const float* in, float* out, int n);

// A/B/C: 스레드 하나가 값 하나
static int RunOnePerThread(const float* in, float* out, int n,
                           void (*k)(const float*, float*, int))
{
    const int blocks = DivUp(n, BLOCK_SIZE);
    k<<<blocks, BLOCK_SIZE>>>(in, out, n);
    CUDA_CHECK_LAUNCH();
    return blocks;
}

static int RunA(const float* in, float* out, int n)
{
    return RunOnePerThread(in, out, n, ReduceDivergent);
}

static int RunB(const float* in, float* out, int n)
{
    return RunOnePerThread(in, out, n, ReduceNoDivergent);
}

static int RunC(const float* in, float* out, int n)
{
    return RunOnePerThread(in, out, n, ReduceSequential);
}

// D/E: 블록 수를 상한으로 묶고 나머지는 grid-stride 로
// 스레드당 ELEMS_PER_THREAD 칸을 목표로 블록 수를 잡고, CAP 에서 자른다.
// 자르고 남은 일은 커널 안 grid-stride 루프가 알아서 더 가져간다.
// 블록 1개여도 그 루프 덕에 n 이 얼마든 처리된다 -> 마지막 패스가 한 번에 끝난다.
static int GridStrideBlocks(int n)
{
    const int want = DivUp(n, BLOCK_SIZE * ELEMS_PER_THREAD);
    if (want < 1)
    {
        return 1;
    }
    if (want < GRID_STRIDE_CAP)
    {
        return want;
    }
    return GRID_STRIDE_CAP;
}

static int RunD(const float* in, float* out, int n)
{
    const int blocks = GridStrideBlocks(n);
    ReduceGridStride<<<blocks, BLOCK_SIZE>>>(in, out, n);
    CUDA_CHECK_LAUNCH();
    return blocks;
}

static int RunE(const float* in, float* out, int n)
{
    const int blocks = GridStrideBlocks(n);
    ReduceShuffle<<<blocks, BLOCK_SIZE>>>(in, out, n);
    CUDA_CHECK_LAUNCH();
    return blocks;
}

// F: 누적 칸을 0 으로 지우고 한 번만 런치. 항상 출력 1개.
// memset 도 측정 안에 포함된다 — atomic 방식의 실제 비용이니까.
static int RunF(const float* in, float* out, int n)
{
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), 0));
    const int blocks = GridStrideBlocks(n);
    ReduceShuffleAtomic<<<blocks, BLOCK_SIZE>>>(in, out, n);
    CUDA_CHECK_LAUNCH();
    return 1;
}

// ---------------------------------------------------------------- 다중 패스 드라이버
// 출력이 1개가 될 때까지 같은 커널을 반복한다. 버퍼 두 개를 번갈아 쓴다(핑퐁).
// 중간에 동기화나 memcpy 를 넣지 않는다 — 넣으면 측정에 CPU 대기가 섞인다.
struct PassResult
{
    const float* result;   // 최종 스칼라가 들어 있는 디바이스 포인터
    int passes;
};

static PassResult LaunchUntilOne(RunFn run, const float* dIn, int n,
                                 float* bufA, float* bufB)
{
    const float* src = dIn;
    float* dst = bufA;
    int cur = n;
    int passes = 0;

    while (true)
    {
        const int produced = run(src, dst, cur);
        ++passes;
        if (produced <= 1)
        {
            break;
        }
        cur = produced;
        src = dst;
        dst = (dst == bufA) ? bufB : bufA;
    }

    PassResult pr;
    pr.result = dst;
    pr.passes = passes;
    return pr;
}

// ---------------------------------------------------------------- CPU 기준
// 정답지: double 로 누적한다. float 입력 1678만개면 double 유효자리(53비트)로도
// 충분히 여유가 있어 사실상 정확하다.
static double SumRefDouble(const std::vector<float>& v)
{
    double s = 0.0;
    for (float x : v)
    {
        s += static_cast<double>(x);
    }
    return s;
}

// float 로 앞에서부터 순차 누적. 흔히 쓰는 방식이고, 여기서 무너진다.
// 합이 커지면(예: 800만) 거기에 0.5 를 더할 때 float 유효자리 24비트가 모자라
// 작은 값이 통째로 버려진다. 이게 1678만번 쌓인다.
static float SumCpuFloatSeq(const std::vector<float>& v)
{
    float s = 0.0f;
    for (float x : v)
    {
        s += x;
    }
    return s;
}

// Kahan 보정합: 매번 버려진 오차를 c 에 담아 다음 덧셈에 되돌린다.
// float 만 쓰면서도 정확도를 크게 끌어올린다 (대신 덧셈이 4배).
static float SumCpuKahan(const std::vector<float>& v)
{
    float s = 0.0f;
    float c = 0.0f;
    for (float x : v)
    {
        const float y = x - c;
        const float t = s + y;
        c = (t - s) - y;      // 이번에 버려진 몫
        s = t;
    }
    return s;
}

static double RelError(double got, double ref)
{
    if (ref == 0.0)
    {
        return 0.0;
    }
    return std::fabs(got - ref) / std::fabs(ref);
}

// 리덕션은 입력만 읽고 출력은 거의 없다 -> 읽은 바이트 / 시간
static double BandwidthGBs(size_t elems, double ms)
{
    const double bytes = static_cast<double>(elems) * sizeof(float);
    return bytes / (ms * 1.0e6);
}

struct Measured
{
    double ms;
    double sum;
    int passes;
};

static Measured MeasureKernel(RunFn run, const float* dIn, int n,
                              float* bufA, float* bufB, int repeat)
{
    // 워밍업. 첫 런치엔 컨텍스트/JIT 비용이 섞인다.
    PassResult pr = LaunchUntilOne(run, dIn, n, bufA, bufB);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    std::vector<double> samples;
    samples.reserve(repeat);
    for (int r = 0; r < repeat; ++r)
    {
        t.Start();
        pr = LaunchUntilOne(run, dIn, n, bufA, bufB);
        t.Stop();
        samples.push_back(t.ElapsedMs());
    }

    float host = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host, pr.result, sizeof(float), cudaMemcpyDeviceToHost));

    Measured m;
    m.ms     = Median(samples);
    m.sum    = static_cast<double>(host);
    m.passes = pr.passes;
    return m;
}

int main(int argc, char** argv)
{
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지

    const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    if (n <= 0)
    {
        std::printf("N 은 1 이상이어야 합니다.\n");
        return 1;
    }

    const size_t elems = static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);
    const int kRepeat = 15;

    std::printf("=== 06_Reduction  N = %d  (%.1f MB) ===\n\n", n, bytes / 1048576.0);
    // 주의: nvcc 프론트엔드는 /utf-8 을 못 받는다. 문자열 끝을 한글로 두면
    //       바로 뒤 \n 이 깨져서 literal 로 찍힌다 -> 항상 ASCII 로 끝낼 것.
    std::printf("블록 %d threads, grid-stride 블록 상한 %d\n\n", BLOCK_SIZE, GRID_STRIDE_CAP);

    // ---- 호스트 데이터. [0,1) 난수를 쓴다.
    // 전부 1.0f 같은 값으로 채우면 어떤 순서로 더해도 답이 같아서
    // 부동소수점 오차 실험이 무의미해진다. 값이 작고 개수가 많을수록
    // "큰 누적값 + 작은 값" 상황이 반복되어 순차합이 무너진다.
    std::vector<float> hIn(elems);
    unsigned rng = 12345u;          // 고정 시드 (실행마다 같은 입력)
    for (size_t i = 0; i < elems; ++i)
    {
        rng = rng * 1664525u + 1013904223u;                  // 전형적인 LCG
        hIn[i] = static_cast<float>(rng >> 8) / 16777216.0f;  // 상위 24비트 -> [0,1)
    }

    // ---- CPU 기준값 3종
    std::printf("CPU 기준 합계 계산 중...");
    std::fflush(stdout);
    const double refSum = SumRefDouble(hIn);

    CpuTimer cpu;
    cpu.Start();
    const float cpuSeq = SumCpuFloatSeq(hIn);
    const double cpuSeqMs = cpu.ElapsedMs();

    cpu.Start();
    const float cpuKahan = SumCpuKahan(hIn);
    const double cpuKahanMs = cpu.ElapsedMs();
    std::printf(" done\n\n");

    // ---- 디바이스 메모리
    // 중간 버퍼는 첫 패스 출력 개수(최대 DivUp(n, BLOCK_SIZE))만큼 있으면 된다.
    float* dIn   = nullptr;
    float* dBufA = nullptr;
    float* dBufB = nullptr;
    const size_t partialCount = static_cast<size_t>(DivUp(n, BLOCK_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&dIn,   bytes));
    CUDA_CHECK(cudaMalloc((void**)&dBufA, partialCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dBufB, partialCount * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    struct Row
    {
        const char* name;
        RunFn run;
    };
    const Row kernels[] = {
        {"A) Divergent",      RunA},
        {"B) NoDivergent",    RunB},
        {"C) Sequential",     RunC},
        {"D) GridStride",     RunD},
        {"E) Shuffle",        RunE},
        {"F) Shuffle+atomic", RunF},
    };
    const int kCount = static_cast<int>(sizeof(kernels) / sizeof(kernels[0]));

    Measured got[6];
    for (int i = 0; i < kCount; ++i)
    {
        got[i] = MeasureKernel(kernels[i].run, dIn, n, dBufA, dBufB, kRepeat);
    }

    // ---- 표
    std::printf("| 커널 | 패스 | kernel ms | 실효 대역폭 | A 대비 | 합계 | 상대오차 |\n");
    std::printf("|---|---:|---:|---:|---:|---:|---:|\n");
    for (int i = 0; i < kCount; ++i)
    {
        std::printf("| %-17s | %4d | %8.3f | %7.1f GB/s | %5.2fx | %.4f | %.2e |\n",
                    kernels[i].name, got[i].passes, got[i].ms,
                    BandwidthGBs(elems, got[i].ms),
                    got[0].ms / got[i].ms,
                    got[i].sum, RelError(got[i].sum, refSum));
    }

    // ---- 부동소수점 비교
    std::printf("\n| 합계 방법 | 시간 | 합계 | 상대오차 |\n");
    std::printf("|---|---:|---:|---:|\n");
    std::printf("| CPU double (ref)   | %8s | %.4f | %.2e |\n",
                "-", refSum, 0.0);
    std::printf("| CPU float seq      | %6.1f ms | %.4f | %.2e |\n",
                cpuSeqMs, static_cast<double>(cpuSeq), RelError(cpuSeq, refSum));
    std::printf("| CPU float Kahan    | %6.1f ms | %.4f | %.2e |\n",
                cpuKahanMs, static_cast<double>(cpuKahan), RelError(cpuKahan, refSum));
    std::printf("| GPU float tree (E) | %6.3f ms | %.4f | %.2e |\n",
                got[4].ms, got[4].sum, RelError(got[4].sum, refSum));

    // printf 문자열은 반드시 ASCII 로 끝낸다. 한글 바로 뒤의 \n 은 literal 로 깨진다
    // (-Xcompiler "/utf-8" 은 cl 에만 가고 nvcc 프론트엔드엔 안 간다)
    std::printf("\n[해석 거리 - README 에 적어볼 것]\n");
    std::printf("  - GPU 트리합이 CPU 순차합보다 정확한 이유 (?)\n");
    std::printf("  - F) 를 여러 번 실행하면 합계 끝자리가 흔들리는 이유 (?)\n");
    std::printf("  - CPU 순차 %.1f ms  ->  E 대비 %.0fx\n", cpuSeqMs, cpuSeqMs / got[4].ms);

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dBufA));
    CUDA_CHECK(cudaFree(dBufB));
    return 0;
}
