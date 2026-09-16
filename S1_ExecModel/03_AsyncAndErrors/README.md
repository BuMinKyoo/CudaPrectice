# 03_AsyncAndErrors — 비동기 런치와 에러 처리 (S1-3)

## 배울 것
- 커널 런치는 **비동기**: `Kernel<<<...>>>()` 는 GPU 큐에 넣고 바로 돌아온다
- 그래서 CPU 스톱워치로 런치 전후를 재면 틀린다 → `cudaEventRecord` / `cudaEventElapsedTime`
- `cudaDeviceSynchronize()` 가 필요한 곳 (측정, 결과 확인) 과 쓰면 안 되는 곳 (실시간 루프 — S7)
- 런치 에러는 반환값이 없다 → `cudaGetLastError()` (꺼내면서 지움) / `cudaPeekAtLastError()` (보기만)
- 런치 설정 에러(non-sticky) vs 실행 중 에러(sticky, 컨텍스트 사망)의 차이

## 결과 (출력 붙여넣기)

```
[A]
(1) 런치 직후 CPU 시간            :
(2) cudaDeviceSynchronize 후 CPU  :
(3) GpuTimer (cudaEvent)          :

[B]
Peek #1 :
Get  #1 :
Get  #2 :
```

## 해석 (직접 적기)
- (1)과 (3)이 몇 배 차이 났나:
- CPU 작업이 GPU 작업과 겹쳤다는 증거:
- `CUDA_CHECK_LAUNCH()` 를 빼면 에러가 어디서 튀어나오게 되나:

## 규칙 (앞으로 모든 예제에 적용)
1. 모든 CUDA API 는 `CUDA_CHECK(...)` 로 감싼다.
2. 모든 커널 런치 뒤엔 `CUDA_CHECK_LAUNCH()`.
3. 시간은 `GpuTimer` 로, 반복 측정 후 `Median`.
