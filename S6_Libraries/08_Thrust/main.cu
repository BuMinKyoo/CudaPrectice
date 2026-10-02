// 08_Thrust — 직접 짤 것인가, 라이브러리를 쓸 것인가
//
// 05 에서 리덕션을 직접 짰다. 경쟁 상태를 보고, 트리로 접고, 공유 메모리를 쓰고,
// atomicAdd 호출을 2048분의 1로 줄였다. 그 과정에서 많은 것을 배웠다.
//
// 그런데 실무에서 합을 구할 때 그 코드를 쓸 것인가? 아니다.
//
//   thrust::reduce(thrust::device, ptr, ptr + n, 0.0f);
//
// 한 줄이다. 그리고 아마 더 빠르다.
//
// 이것이 모순이 아니다.
//   05 를 직접 짠 이유 = 이해하기 위해
//   실무에서 쓸 것      = 라이브러리
// GPU 가 어떻게 도는지 모르면 라이브러리가 왜 빠른지도, 언제 느린지도 모른다.
// 그래서 05 를 먼저 하고 08 을 나중에 한다.
//
// 이 예제가 비교하는 것
//   1) 05 의 segmented 커널을 그대로 가져온 것   (직접 짬, 약 30줄 + 설정)
//   2) thrust::reduce                           (1줄)
//   3) thrust::transform_reduce  제곱합          (1줄. 직접 짜려면 커널을 또 고쳐야 한다)
//   4) thrust::sort                             (1줄. 직접 짜려면 며칠이다)
//
// 사용법:  08_Thrust.exe [size]     기본 3355만 (05 와 같은 크기)

#include "cu_common.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------- 05 에서 가져온 커널
// 05_Reduction 의 segmentedSumReductionKernel 과 같다.
// 블럭마다 구간을 맡아 공유 메모리에서 트리로 접고, 블럭당 atomicAdd 한 번.
__global__ void SegmentedSum(const float* input, float* output) {
    extern __shared__ float s[];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    s[t] = input[i] + input[i + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            s[t] += s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, s[0]);
    }
}

// thrust 에 넘길 연산. 람다를 쓰려면 --extended-lambda 플래그가 필요해서
// 구조체로 쓴다. __host__ __device__ 둘 다 붙여야 CPU/GPU 양쪽에서 쓸 수 있다.
struct Square {
    __host__ __device__ float operator()(const float& x) const { return x * x; }
};

// ---------------------------------------------------------------- 시간 측정
// 05 와 달리 반복 측정 + 중앙값을 쓴다 (레포 코드 규칙 3).
template <typename Fn>
static double TimeIt(Fn fn, int repeat = 7) {
    fn();                                      // 워밍업
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeat);
    CpuTimer t;
    for (int r = 0; r < repeat; ++r) {
        t.Start();
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        samples.push_back(t.ElapsedMs());
    }
    return Median(samples);
}

int main(int argc, char** argv) {
    EnableUtf8Console();          // 콘솔 한글 깨짐 방지

    const int size = (argc > 1) ? std::atoi(argv[1]) : 1024 * 1024 * 32;
    const int threadsPerBlock = 1024;
    const int oneBlockSize = threadsPerBlock * 2;

    if (size % oneBlockSize != 0) {
        std::printf("size must be a multiple of %d\n", oneBlockSize);
        return EXIT_FAILURE;
    }

    std::printf("=== 08_Thrust  size = %d (%.0f MB) ===\n\n",
                size, size * sizeof(float) / 1048576.0);

    // ---- 입력
    std::vector<float> h(size);
    std::srand(2025);
    for (int i = 0; i < size; ++i) {
        h[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // 정답지는 double 로. float 로 3355만번 누적하면 그 자체가 부정확하다 (05 참고)
    double ref = 0.0;
    for (int i = 0; i < size; ++i) {
        ref += static_cast<double>(h[i]);
    }
    double refSq = 0.0;
    for (int i = 0; i < size; ++i) {
        refSq += static_cast<double>(h[i]) * h[i];
    }

    float* dIn = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dIn, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dOut, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, h.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    // raw 포인터를 thrust 가 알아볼 수 있게 감싼다.
    // device_vector 를 쓰면 메모리 관리까지 맡길 수 있지만, 여기서는 같은 데이터로
    // 비교해야 하므로 기존 cudaMalloc 포인터를 그대로 쓴다.
    thrust::device_ptr<float> p(dIn);

    std::printf("| 방법 | 시간 | 상대오차 | 코드 |\n");
    std::printf("|---|---:|---:|---|\n");

    // ---- 1) 직접 짠 것 (05 의 결과물)
    float sumMine = 0.0f;
    const double msMine = TimeIt([&]() {
        CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));
        SegmentedSum<<<size / oneBlockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(float)>>>(dIn, dOut);
        CUDA_CHECK_LAUNCH();
    });
    CUDA_CHECK(cudaMemcpy(&sumMine, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("| 05 segmented (직접) | %7.3f ms | %.2e | 커널 30줄 + 런치 설정 |\n",
                msMine, std::fabs(sumMine - ref) / ref);

    // ---- 2) thrust::reduce
    float sumThrust = 0.0f;
    const double msThrust = TimeIt([&]() {
        sumThrust = thrust::reduce(thrust::device, p, p + size, 0.0f);
    });
    std::printf("| thrust::reduce | %7.3f ms | %.2e | 1줄 |\n",
                msThrust, std::fabs(sumThrust - ref) / ref);

    // ---- 3) 제곱합. 직접 짜려면 커널 안의 덧셈을 고쳐야 한다.
    //         thrust 는 "무엇을 변환하고 무엇으로 접을지" 만 바꿔 끼우면 된다.
    float sumSq = 0.0f;
    const double msSq = TimeIt([&]() {
        sumSq = thrust::transform_reduce(thrust::device, p, p + size, Square(), 0.0f,
                                         thrust::plus<float>());
    });
    std::printf("| thrust::transform_reduce (제곱합) | %7.3f ms | %.2e | 1줄 |\n",
                msSq, std::fabs(sumSq - refSq) / refSq);

    // ---- 4) 최댓값. 05 의 커널로 하려면 atomicAdd 를 atomicMax 로 바꿔야 하는데,
    //         float atomicMax 는 하드웨어에 없어서 직접 만들어야 한다 (atomicCAS 루프).
    float maxVal = 0.0f;
    const double msMax = TimeIt([&]() {
        thrust::device_ptr<float> it = thrust::max_element(thrust::device, p, p + size);
        maxVal = *it;
    });
    std::printf("| thrust::max_element | %7.3f ms | %s | 1줄 |\n",
                msMax, maxVal <= 1.0f && maxVal > 0.99f ? "OK" : "??");

    // ---- 5) 정렬. 직접 짜려면 며칠이다 (bitonic / radix sort).
    //         데이터를 망가뜨리므로 마지막에 한다.
    const double msSort = TimeIt(
        [&]() {
            CUDA_CHECK(cudaMemcpy(dIn, h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
            thrust::sort(thrust::device, p, p + size);
        },
        3);
    std::printf("| thrust::sort (복사 포함) | %7.3f ms | - | 1줄 |\n", msSort);

    std::printf("\n[읽는 법]\n");
    std::printf("  05 segmented 와 thrust::reduce 가 하는 일은 똑같다.\n");
    std::printf("  thrust 가 더 빠르다면, 그 안에 들어간 최적화 때문이다:\n");
    std::printf("    - 쓰레드당 여러 칸 처리 (coarsening)\n");
    std::printf("    - stride < 32 부터는 warp shuffle 로 __syncthreads() 없이 접기\n");
    std::printf("    - 입력 크기와 GPU 에 맞춰 블럭/쓰레드 수를 자동 선택\n");
    std::printf("  05 에서 '아직 남은 비효율' 이라고 적어둔 것들이 전부 들어가 있다.\n\n");

    std::printf("[그래서 언제 직접 짜나]\n");
    std::printf("  라이브러리를 쓴다 : 합, 최대/최소, 정렬, 스캔, 행렬곱, FFT 같은\n");
    std::printf("                     이름이 붙은 표준 연산. 이미 누가 몇 년을 갈아 넣었다.\n");
    std::printf("  직접 짠다        : 02 의 BGR->gray 처럼 내 문제에만 있는 연산,\n");
    std::printf("                     여러 단계를 한 커널로 합쳐 메모리 왕복을 줄일 때,\n");
    std::printf("                     라이브러리에 없는 자료구조를 다룰 때.\n\n");

    std::printf("[알아둘 라이브러리]\n");
    std::printf("  Thrust   STL 같은 인터페이스. reduce/sort/scan/transform. 툴킷에 포함\n");
    std::printf("  CUB      Thrust 내부에서 쓰는 저수준 블럭 단위 primitive. 더 빠르고 더 번거롭다\n");
    std::printf("  cuBLAS   행렬/벡터 연산. 행렬곱은 직접 짜면 거의 항상 손해.\n");
    std::printf("  cuFFT    FFT (푸리에 변환)\n");
    std::printf("  cuDNN    딥러닝 전용 (별도 설치)\n");
    std::printf("  CCCL     요즘은 Thrust + CUB + libcudacxx 를 묶어 이렇게 부른다\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    return EXIT_SUCCESS;
}
