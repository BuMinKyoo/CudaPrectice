// 05_Reduction — S3 리덕션: N개를 1개로 접기
//
// 교재: "Programming Massively Parallel Processors: A Hands-on Approach" 4th
// 참고 자료
// - https://github.com/umfranzw/cuda-reduction-example/tree/master/reduce0
// - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//
// 04/05 는 "N개 입력 -> N개 출력" 이었다. 스레드마다 자기 칸이 따로 있어서
// 서로 간섭할 일이 없었다. 리덕션은 다르다. 출력이 하나다.
// 여러 쓰레드가 같은 칸에 기여해야 하니 처음으로 "협력"이 필요해진다.
//
// 커널 (교재 순서, 0 은 비교용)
//   0) racySumReductionKernel        atomicAdd 없이 += . 답이 틀린다 (경쟁 상태)
//   1) atomicSumReductionKernel      쓰레드마다 atomicAdd. 정답은 맞지만 느리다
//   2) convergentSumReductionKernel  블럭 안에서 트리로 접는다 (블럭 하나만)
//   3) sharedMemorySumReductionKernel  전역 대신 공유 메모리에서 접는다 (블럭 하나만)
//   4) segmentedSumReductionKernel   2)+3) 을 합쳐 여러 블럭으로 (실전형)
//
// 2), 3) 은 블럭 하나만 쓰므로 배열 앞쪽 2 * threadsPerBlock 개만 더한다.
// 전체 배열을 더하는 건 1) 과 4) 뿐이다 -> 비교 기준도 따로 쓴다.
//
// 그리고 이 예제에는 04 에 없던 함정이 하나 더 있다.
// 04 는 "mismatch == 0" 이 정답이었다. 값을 옮기기만 했으니 비트가 같아야 했다.
// 리덕션은 더한다. float 덧셈은 결합법칙이 성립하지 않는다.
//   (a+b)+c != a+(b+c)
// 더하는 순서가 바뀌면 답이 바뀐다 -> 비트 일치가 아니라 상대오차로 검증해야 한다.

#include "cu_common.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// 한 번 실행하면서 CPU 시간과 GPU 시간을 같이 잰다.
// 커널 런치는 비동기라 두 값이 다르게 나온다 -> 03_AsyncAndErrors 에서 본 그것.
//
// 주의: 이 예제는 repo 코드 규칙 3(워밍업 + 중앙값)을 일부러 따르지 않는다.
//       교재 코드 형태를 유지하려고 1회 측정만 한다. 대신 main 에서 워밍업을
//       한 번 돌려서 첫 런치의 컨텍스트 생성 비용이 표에 섞이지 않게 했다.
//       그래도 실행마다 편차가 꽤 있으니 몇 번 돌려보고 판단할 것.
void timedRun(const string name, const function<void()> &func) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto startCpu = chrono::high_resolution_clock::now(); // CPU 시간측정 시작
    CUDA_CHECK(cudaEventRecord(start, 0));                // GPU 시간측정 시작

    func(); // 실행

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));             // GPU 시간측정 종료
    auto endCpu = chrono::high_resolution_clock::now(); // CPU 시간측정 종료

    float elapsedGpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedGpu, start, stop));
    chrono::duration<float, milli> elapsedCpu = endCpu - startCpu;
    cout << name << ": CPU " << elapsedCpu.count() << " ms, GPU " << elapsedGpu << " ms" << endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ---------------------------------------------------------------- 0) 경쟁 상태 (틀린 코드)
// 아래 1) 의 주석에 설명된 "하면 안 되는 것" 을 실제로 해본다.
// atomicAdd 없이 그냥 += 로 누적한다. 돌아가고, 죽지도 않고, 답만 틀린다.
//
// 읽기-더하기-쓰기가 세 단계로 쪼개져 있어서, 내가 읽고 쓰는 사이에 다른 쓰레드가
// 끼어들면 그 쓰레드의 기여가 통째로 사라진다. 3355만 개가 동시에 덤비니
// 거의 다 사라지고 합계가 정답의 극히 일부만 나온다.
//
// 보시면 안다 — 실행할 때마 결과가 달라진다. 입력은 고정 시드라 똑같은데도.
// 재현되지 않는 버그가 가장 잡기 어렵다 (-> S4 의 compute-sanitizer)
__global__ void racySumReductionKernel(float *input, float *output) {

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    output[0] += input[i]; // <- 여러 쓰레드가 경쟁적으로 메모리에 접근하기 때문에 오류 발생
}

// ---------------------------------------------------------------- 1) atomic
__global__ void atomicSumReductionKernel(float *input, float *output) {

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // output[0] += input[i]; // <- 여러 쓰레드가 경쟁적으로 메모리에 접근하기 때문에 오류 발생
    //                           (위의 racySumReductionKernel 이 바로 그것이다)

    /*
    * 두 개의 쓰레드가 한 메모리 공간을 두고 경쟁(racing) 하는 사례
    * 원래 값 1에다가 쓰레드 2개가 각각 1씩 더해서 3이 되어야 하는 경우

     1. 시리얼로 하나씩 더할 때
     - 처음에 저장되어 있는 값 output[0] = 1
     - thread 0이 output[0]의 1을 읽어옴
     - thread 0이 읽어온 1에다가 1을 더함
     - thread 0이 output[0]에다가 2를 저장
     - thread 1이 output[0]의 2를 읽어옴
     - thread 1이 읽어온 2에다가 1을 더함
     - thread 1이 output[0]에다가 3을 저장
     - 결과적으로 output[0]은 3

     2. 멀티쓰레딩으로 하나씩 더할 때 문제가 생기는 경우
     - output[0] = 1
     - thread 0: 1 read
     - thread 1: 1 read (thread 0이 write 하기 전에 thread 1이 읽어옴)
     - thread 0: output[0] <- 1 + 1 저장
     - thread 1: output[0] <- 1 + 1 저장 (앞서 thread 0이 저장한 output[0]은 2이지만 thread 1은 알지
    못함)

    멀티쓰레딩을 사용하면 경우에 따라서 문제가 생기지 않을 수도 있습니다. 그래서 정상작동하는 것으로
    착각하는 경우도 많습니다. 내가 구현하는 방식에 메모리 접근 경쟁이 생기는지 아닌지를 항상
    주의해야 합니다.
    */

    // atomicAdd()로 정확하게 계산하지만 지나치게 느려집니다.
    // 여러개의 쓰레드들이 자기 차례를 기다려야 하기 때문입니다.
    //
    // atomicAdd(p, v) 는 읽기-더하기-쓰기를 쪼갤 수 없는 한 덩어리로 처리한다.
    // 그래서 답은 맞다. 대신 3355만 개의 쓰레드가 "한 칸"을 두고 줄을 선다.
    // 이 커널의 측정값이 곧 atomic 경합(contention) 의 비용이다.
    atomicAdd(output, input[i]);
}

// ---------------------------------------------------------------- 2) convergent
// 블럭이 하나일 때만 사용 가능
// 엔비디아 슬라이드 그림 속의 Values는 여기서 input 입니다.
// for문 안의 각 단계에서 입력 배열(input)에 덮어쓰면서 reduce 해 나가다가
// 마지막 하나만 output에 저장합니다.
//
// convergent = "일하는 쓰레드가 앞쪽으로 모인다".
//   stride=1024: t0 += t1024, t1 += t1025, ... t1023 += t2047
//   stride=512 : t0 += t512,  t1 += t513,  ... t511  += t1023
//   stride=256 : ...
// 일하는 쓰레드가 항상 threadIdx.x < stride 라 앞쪽에 연속으로 모인다.
// warp(32개) 단위로 통째로 일하거나 통째로 쉬므로 warp divergence 가 없다.
// (반대로 "옆칸 흡수" 방식은 짝수 tid 만 일해서 warp 안이 갈라진다 -> 느리다)
//
// 주의: input 을 덮어쓴다. 같은 배열로 다른 커널을 또 돌리려면 다시 채워야 한다.
__global__ void convergentSumReductionKernel(float *input,
                                             float *output) { // block 하나로 처리가능한 크기
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            // i를 사용해도 되고 threadIdx.x를 직접 사용해도 됩니다.
            input[i] += input[i + stride];
        }
        __syncthreads(); // <- 같은 블럭 안에 있는 쓰레드들 동기화
    }
    if (threadIdx.x == 0)
        *output = input[0];
}

// ---------------------------------------------------------------- 3) shared memory
// 2) 는 단계마다 전역 메모리를 읽고 썼다. 10단계면 전역 왕복이 10번이다.
// 공유 메모리는 SM 안에 있어서 전역 메모리보다 훨씬 빠르다.
// 처음에 한 번만 전역에서 읽어 공유 메모리에 올리고, 접는 건 전부 거기서 한다.
__global__ void sharedMemorySumReductionKernel(float *input, float *output) {

    // __shared__ float inputShared[1024];

    extern __shared__ float inputShared[]; // <- 블럭 안에서 여러 쓰레드들이 공유하는 빠른 메모리
    // extern + 크기 없음 = 런치할 때 세 번째 인자로 크기를 넘긴다
    //   kernel<<<블럭수, 쓰레드수, 공유메모리크기>>>(...)

    unsigned int t = threadIdx.x;

    // 전역에서 읽어오면서 이미 한 번 접는다 (2048개 -> 1024개)
    inputShared[t] = input[t] + input[t + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {

        // __syncthreads() 가 루프 맨 앞에 있는 이유:
        // 첫 바퀴에서는 위의 inputShared 쓰기가 전부 끝나기를 기다리고,
        // 두 번째 바퀴부터는 직전 바퀴의 쓰기가 끝나기를 기다린다. 한 줄로 둘 다 된다.
        //
        // 그리고 이 장벽은 반드시 if 밖에 있어야 한다. 블럭 안 "모든" 쓰레드가
        // 도달해야 통과하는 장벽이라, 일부만 들어가는 if 안에 넣으면 나머지는
        // 영원히 기다린다 (deadlock).
        __syncthreads();

        if (threadIdx.x < stride) {
            inputShared[t] += inputShared[t + stride];
        }
    }
    if (t == 0)
        *output = inputShared[0];
}

// ---------------------------------------------------------------- 4) segmented
// 2)+3) 을 합치면 블럭 제한이 풀린다.
// 블럭 하나가 배열의 한 구간(segment = 2 * blockDim 칸)을 맡아 공유 메모리에서 접고,
// 블럭별 결과 하나씩만 atomicAdd 로 합친다.
//
// 1) 과 비교하면 atomicAdd 호출이
//   1) size 번        = 3355만 번
//   4) size/2048 번   = 1만 6384 번    -> 2048배 감소
// 이 숫자 하나가 두 커널의 시간 차이를 거의 다 설명한다.
//
// 읽는 순서도 중요하다. i = segment + threadIdx.x 라서 한 warp 32개가
// 연속된 32칸을 읽는다 (coalescing, 04 에서 본 그것).
__global__ void segmentedSumReductionKernel(float *input, float *output) {
    extern __shared__ float inputShared[];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // 위의 두 개를 잘 합치면 됩니다.
    inputShared[t] = input[i] + input[i + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            inputShared[t] += inputShared[t + stride];
        }
    }

    // 블럭당 딱 한 번만 부르는 게 요점이다. 쓰레드마다 부르면 1) 이 된다.
    if (t == 0)
        atomicAdd(output, inputShared[0]);
}

int main(int argc, char *argv[]) {

    EnableUtf8Console(); // 콘솔 한글 깨짐 방지

    const int size = 1024 * 1024 * 32;
    int threadsPerBlock = 1024;

    // 2), 3) 이 맡는 범위. 블럭 하나가 쓰레드당 2칸씩 = 2048칸
    const int oneBlockSize = threadsPerBlock * 2;

    if (size % oneBlockSize != 0) {
        cout << "size must be a multiple of " << oneBlockSize << endl;
        return EXIT_FAILURE;
    }

    cout << "=== 05_Reduction  size = " << size << " (" << (size * sizeof(float) / 1048576)
         << " MB) ===" << endl;
    cout << "threadsPerBlock = " << threadsPerBlock << ", 1-block range = " << oneBlockSize << endl
         << endl;

    // 배열 만들기
    vector<float> arr(size);
    srand(2025); // 고정 시드. 실행마다 같은 입력이어야 오차를 비교할 수 있다
    for (int i = 0; i < size; i++)
        arr[i] = (float)rand() / RAND_MAX;

    // CPU에서 합 구하기
    float sumCpu = 0.0f;
    timedRun("CPU Sum", [&]() {
        for (int i = 0; i < size; i++) {
            sumCpu += arr[i];
        }
    });

    // 정답지. float 로 3355만 번 누적하면 sumCpu 자체가 이미 부정확하다.
    // 누적값이 1600만쯤 됐을 때 거기에 0.5 를 더하면 float 유효자리 24비트가
    // 모자라서 작은 값이 통째로 버려지기 때문이다. 그래서 double 로 따로 구한다.
    double sumRef = 0.0;
    for (int i = 0; i < size; i++)
        sumRef += (double)arr[i];

    // 2), 3) 은 앞쪽 2048개만 더하므로 비교 기준도 따로 필요하다
    double sumRefOneBlock = 0.0;
    for (int i = 0; i < oneBlockSize; i++)
        sumRefOneBlock += (double)arr[i];

    // GPU 준비
    float *dev_input;
    float *dev_output;

    CUDA_CHECK(cudaMalloc(&dev_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_input, arr.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    // cudaMalloc 은 0으로 초기화해주지 않는다. 1)/4) 는 atomicAdd 로 "누적" 하므로
    // 지우지 않으면 쓰레기값 위에 더해진다. 커널 돌리기 전에 매번 지운다.
    auto clearOutput = [&]() { CUDA_CHECK(cudaMemset(dev_output, 0, sizeof(float))); };

    // 2) 는 input 을 덮어쓴다. 그래서 앞쪽 2048칸을 원본으로 되돌려 놔야
    // 뒤에 오는 3), 4) 가 제대로 된 값을 읽는다.
    auto restoreHead = [&]() {
        CUDA_CHECK(cudaMemcpy(dev_input, arr.data(), oneBlockSize * sizeof(float),
                              cudaMemcpyHostToDevice));
    };

    float sumGpu = 0.0f;
    auto readResult = [&]() {
        CUDA_CHECK(cudaMemcpy(&sumGpu, dev_output, sizeof(float),
                              cudaMemcpyDeviceToHost)); // 숫자 하나만 복사
        return (double)sumGpu;
    };

    // cout 의 서식 플래그는 한 번 바꾸면 계속 남는다. 여기서 scientific 을 켜둔 채
    // 나가면 timedRun 의 ms 까지 지수 표기로 찍힌다 -> 끝에 defaultfloat 로 되돌린다.
    auto report = [&](double got, double ref) {
        cout << "    sum = " << fixed << setprecision(4) << got << ", ref = " << ref
             << ", rel.err = " << scientific << setprecision(2) << fabs(got - ref) / fabs(ref)
             << defaultfloat << setprecision(6) << endl;
    };

    // 워밍업. 첫 커널 런치에는 컨텍스트 생성 비용이 섞여서 측정이 부풀려진다.
    clearOutput();
    segmentedSumReductionKernel<<<size / oneBlockSize, threadsPerBlock,
                                  threadsPerBlock * sizeof(float)>>>(dev_input, dev_output);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    const int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // ---- 0) 경쟁 상태. atomicAdd 없이 그냥 += 한 버전 (틀린 답이 나온다)
    // 같은 커널을 두 번 돌려서, 입력이 똑같은데도 답이 달라지는 것까지 보여준다.
    clearOutput();
    timedRun("Racy     ", [&]() {
        racySumReductionKernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);
        CUDA_CHECK_LAUNCH();
    });
    const double racy1 = readResult();

    clearOutput();
    racySumReductionKernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    const double racy2 = readResult();

    // 주의: 문자열을 한글로 끝내면 안 된다. nvcc 프론트엔드가 /utf-8 을 못 받아서
    //       한글 마지막 바이트가 바로 뒤 문자를 짝으로 삼아 삼킨다.
    //       \n 이 literal 로 찍히거나, 심하면 닫는 따옴표가 먹혀 컴파일이 깨진다.
    cout << "    sum = " << fixed << setprecision(4) << racy1 << " (1st) / " << racy2
         << " (2nd)  <- WRONG, 실행마다 다름 (!)" << endl;
    cout << "    ref = " << sumRef << "  ->  정답의 " << setprecision(4)
         << 100.0 * racy1 / sumRef << " % 만 살아남았다 (!)" << defaultfloat << setprecision(6)
         << endl;

    // ---- 1) 전체 배열, 쓰레드마다 atomicAdd
    clearOutput();
    timedRun("Atomic   ", [&]() {
        atomicSumReductionKernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);
        CUDA_CHECK_LAUNCH();
    }); // 68 ms
    report(readResult(), sumRef);

    // ---- 2) 블럭 하나. 배열 앞쪽 2048개만 더한다
    restoreHead();
    clearOutput();
    timedRun("Convergent", [&]() {
        convergentSumReductionKernel<<<1, threadsPerBlock>>>(dev_input,
                                                             dev_output); // 블럭이 하나일 때만 사용
        CUDA_CHECK_LAUNCH();
    });
    report(readResult(), sumRefOneBlock);

    // ---- 3) 블럭 하나 + 공유 메모리
    restoreHead(); // 2) 가 앞쪽을 덮어썼으므로 되돌린다
    clearOutput();
    timedRun("Shared   ", [&]() {
        sharedMemorySumReductionKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            dev_input, dev_output); // 블럭이 하나일 때만 사용
        CUDA_CHECK_LAUNCH();
    });
    //  kernel<<<블럭수, 쓰레드수, 공유메모리크기>>>(...);
    report(readResult(), sumRefOneBlock);

    // ---- 4) 전체 배열, 여러 블럭 + 공유 메모리 + 블럭당 atomicAdd 1번
    clearOutput();
    timedRun("Segmented", [&]() {
        int numBlocks = size / oneBlockSize; // size 나누기 2 주의
        segmentedSumReductionKernel<<<numBlocks, threadsPerBlock,
                                      threadsPerBlock * sizeof(float)>>>(dev_input, dev_output);
        CUDA_CHECK_LAUNCH();
    }); // 1 ms 근처
    report(readResult(), sumRef);

    cout << endl;
    cout << "sumCpu(float) = " << fixed << setprecision(4) << sumCpu << ", sumRef(double) = "
         << sumRef << endl;
    cout << "CPU float rel.err = " << scientific << setprecision(2)
         << fabs((double)sumCpu - sumRef) / sumRef << defaultfloat << setprecision(6) << endl;

    cout << endl << "[해석 거리 - README 에 적어볼 것]" << endl;
    cout << "  - 0) 이 1) 보다 훨씬 빠르다. 틀린 코드가 빠른 것이 왜 위험한가 (?)" << endl;
    cout << "  - 0) 에서 기여가 사라지는 과정을 쓰레드 2개로 적어볼 것 (커널 주석 참고)" << endl;
    cout << "  - 1) 이 느린 이유. atomicAdd 호출 횟수와 시간이 비례하는가 (?)" << endl;
    cout << "  - 2) -> 3) 은 접는 방식이 같은데 왜 빨라지는가 (?)" << endl;
    cout << "  - 1) 의 오차가 CPU float 순차합과 비슷한 이유 (?)" << endl;
    cout << "  - timedRun 의 CPU 시간과 GPU 시간이 거의 같게 나오는 이유 (?)" << endl;
    cout << "    (커널 런치는 비동기인데 왜? timedRun 안에서 무엇을 하고 있나)" << endl;

    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_output));

    return EXIT_SUCCESS;
}
