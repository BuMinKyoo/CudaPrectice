# 08_Thrust — 직접 짤 것인가, 라이브러리를 쓸 것인가 (S6)

## 왜 이걸 마지막에 하나

05 에서 리덕션을 직접 짰다. 경쟁 상태를 보고, 트리로 접고, 공유 메모리를 쓰고,
`atomicAdd` 호출을 2048분의 1로 줄였다. 꽤 긴 과정이었다.

그런데 실무에서 합을 구할 때 그 코드를 쓸 것인가? **아니다.**

```cpp
thrust::reduce(thrust::device, ptr, ptr + n, 0.0f);
```

한 줄이다. 그리고 **더 빠르다.**

이것이 모순이 아니다.

| | |
|---|---|
| 05 를 직접 짠 이유 | **이해하기 위해** |
| 실무에서 쓸 것 | **라이브러리** |

GPU 가 어떻게 도는지 모르면 **라이브러리가 왜 빠른지도, 언제 느린지도 모른다.**
그래서 05 를 먼저 하고 08 을 나중에 한다. 순서를 바꾸면 "그냥 reduce 쓰면 되네"
에서 끝나고, 왜 그게 빠른지는 영영 모른다.

## 측정 결과 — P2000, 128 MB

| 방법 | 시간 | 상대오차 | 코드 |
|---|---:|---:|---|
| 05 segmented (직접) | 4.566 ms | 6.89e-07 | 커널 30줄 + 런치 설정 |
| **thrust::reduce** | **1.143 ms** | **9.26e-08** | **1줄** |
| thrust::transform_reduce (제곱합) | 1.146 ms | 4.92e-08 | 1줄 |
| thrust::max_element | 1.161 ms | OK | 1줄 |
| thrust::sort (복사 포함) | 55.055 ms | - | 1줄 |

**4배 빠르고 7배 정확하다.** 그리고 30줄이 1줄이 됐다.

### 왜 더 빠른가

05 README 의 "아직 남은 비효율" 에 적어둔 것들이 **전부 들어가 있다.**

| 05 에 없던 것 | 하는 일 |
|---|---|
| 쓰레드당 여러 칸 (coarsening) | 블럭 하나가 2048칸이 아니라 훨씬 많이 맡는다 |
| **warp shuffle** | `stride < 32` 부터는 `__syncthreads()` 없이 warp 안에서 접는다 |
| 자동 튜닝 | 입력 크기와 GPU 에 맞춰 블럭/쓰레드 수를 고른다 |

05 에서 "뒷단계는 1024명 중 1명만 일한다" 고 적은 그 비효율을 warp shuffle 이 없앤다.

### 왜 더 정확한가

더 깊은 트리로 접기 때문이다. 05 는 블럭당 2048칸을 접은 뒤 1만 6천개의 부분합을
`float` 한 칸에 `atomicAdd` 로 모았다. 그 마지막 단계에서 자릿수를 잃는다.

**"빠른 쪽이 정확하기까지 하다"** 는 05 의 결론이 여기서 한 번 더 성립한다.

## thrust 쓰는 법

### 기존 포인터를 감싸기

```cpp
float* dIn;
cudaMalloc(&dIn, n * sizeof(float));
...
thrust::device_ptr<float> p(dIn);              // 감싸기만 한다. 복사 없음
float sum = thrust::reduce(thrust::device, p, p + n, 0.0f);
```

### 아예 thrust 에게 메모리를 맡기기

```cpp
thrust::device_vector<float> d(h.begin(), h.end());   // 할당 + 복사 한 줄
float sum = thrust::reduce(d.begin(), d.end(), 0.0f);
// 소멸자가 cudaFree 까지 해준다
```

`std::vector` 와 거의 같은 인터페이스다. `cudaMalloc`/`cudaMemcpy`/`cudaFree` 를
직접 안 써도 된다.

### 연산 바꿔 끼우기

```cpp
// 합
thrust::reduce(thrust::device, p, p + n, 0.0f);

// 제곱합 — "무엇으로 변환하고 무엇으로 접을지" 만 바꾼다
thrust::transform_reduce(thrust::device, p, p + n, Square(), 0.0f, thrust::plus<float>());

// 최댓값
thrust::max_element(thrust::device, p, p + n);
```

05 의 커널로 제곱합을 하려면 커널 안을 고쳐야 하고, 최댓값을 하려면
`atomicAdd` 를 `atomicMax` 로 바꿔야 하는데 **float atomicMax 는 하드웨어에 없어서
`atomicCAS` 루프로 직접 만들어야 한다.** thrust 는 인자만 바꾼다.

> 람다 대신 구조체(`Square`)를 쓴 이유: device 람다를 쓰려면 `--extended-lambda`
> 컴파일 플래그가 필요하다. 구조체는 그냥 된다.

## 그래서 언제 직접 짜나

| | |
|---|---|
| **라이브러리를 쓴다** | 합, 최대/최소, 정렬, 스캔, 행렬곱, FFT 같은 **이름이 붙은 표준 연산.** 이미 누가 몇 년을 갈아 넣었다 |
| **직접 짠다** | 02 의 BGR→gray 처럼 **내 문제에만 있는 연산** |
| | 여러 단계를 **한 커널로 합쳐** 메모리 왕복을 줄일 때 |
| | 라이브러리에 없는 자료구조를 다룰 때 |

**중간 정도로 자주 쓰는 판단:** 라이브러리 호출을 여러 번 이어 붙이면 그때마다
전역 메모리를 왕복한다. 단계가 많아지면 직접 하나로 합치는 게 이길 수 있다.
04/05 에서 배운 "메모리 왕복을 줄인다" 가 여기서 판단 기준이 된다.

## 알아둘 라이브러리

| | 무엇 |
|---|---|
| **Thrust** | STL 같은 인터페이스. reduce / sort / scan / transform. 툴킷에 포함 |
| **CUB** | Thrust 내부에서 쓰는 저수준 블럭 단위 primitive. 더 빠르고 더 번거롭다 |
| **cuBLAS** | 행렬/벡터 연산. **행렬곱을 직접 짜는 것은 거의 항상 손해다** |
| cuFFT | 푸리에 변환 |
| cuDNN | 딥러닝 전용 (별도 설치) |
| **CCCL** | 요즘은 Thrust + CUB + libcudacxx 를 묶어 이렇게 부른다 |

## 실행

```
08_Thrust.exe [size]       기본 3355만 (05 와 같은 크기)
```

## 결과 (직접 돌려보고 적기)

| 방법 | 시간 | 상대오차 |
|---|---:|---:|
| 05 segmented (직접) | | |
| thrust::reduce | | |
| thrust::transform_reduce | | |
| thrust::max_element | | |
| thrust::sort | | |

## 해석 (직접 적기)

- thrust::reduce 가 05 보다 몇 배 빠른가. 05 에서 "아직 남은 비효율" 로
  적어둔 것 중 어느 것이 그 차이를 만들고 있을 것 같나:
- thrust 가 05 보다 정확하기까지 하다. 왜인가 (05 의 마지막 atomicAdd 단계와 연결):
- `thrust::sort` 가 reduce 보다 50배 느리다. 정렬이 리덕션보다 본질적으로
  어려운 이유는? (리덕션은 N→1, 정렬은 N→N 이면서 모든 원소가 서로 비교된다)
- 내가 지금 만들려는 것 중에 "이름이 붙은 표준 연산" 은 무엇이고
  "내 문제에만 있는 것" 은 무엇인가:

## 생각해볼 것

- `thrust::device_vector` 로 바꿔 쓰면 코드가 얼마나 줄어드나?
- `thrust::reduce` 를 `thrust::host` 로 실행하면? (같은 코드가 CPU 에서 돈다)
- 제곱합을 05 의 커널로 구현하려면 어디를 고쳐야 하나. 몇 줄인가?
- 최댓값을 05 의 커널로 하려면? float `atomicMax` 가 없다는 것을 확인해 볼 것
- `transform` + `reduce` 를 따로 부르는 것과 `transform_reduce` 한 번의 차이는?
  (중간 결과를 전역 메모리에 썼다가 다시 읽느냐 아니냐)
- cuBLAS 로 내적(`cublasSdot`)을 해보고 `transform_reduce` 와 비교해 보기

## 여기까지 왔으면

```
S0  환경           GPU 가 무엇인지 안다
S1  실행 모델       GPU 에서 코드가 돌게 만든다
S2  메모리         느린 이유를 안다
S3  리덕션         쓰레드가 협력하게 만든다
S4  검증           틀린 것을 틀렸다고 증명한다
S5  스트림         복사와 계산을 겹친다
S6  라이브러리      직접 짤지 말지 판단한다
```

**CUDA 기본기는 여기까지다.** 다음은 실제로 만들고 싶은 것을 만들면서
필요한 것만 그때그때 배우면 된다.

남은 것들(scan, warp shuffle, cooperative groups, multi-GPU, 텐서코어)은
**필요해지면 그때 배우는 것**이지, 미리 해둘 것은 아니다.
