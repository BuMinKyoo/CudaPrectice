# 01_VectorAdd — 블록 크기와 grid-stride (S1-1)

## 배울 것
- `<<<grid, block>>>` 런치, 전역 인덱스 `blockIdx.x * blockDim.x + threadIdx.x`
- `if (i < n)` 가드가 왜 필요한가 (grid×block 은 n 보다 크게 반올림됨)
- **warp = 32 스레드** → 블록 크기를 32의 배수로 잡는 이유
- **grid-stride loop**: 스레드 수와 데이터 크기를 분리하는 표준 패턴
- 시간은 `GpuTimer`(cudaEvent)로 잰다 — 이유는 03 예제에서

## 실행
Release | x64 로 빌드 후 실행. `01_VectorAdd.exe 50000000` 처럼 N 을 바꿔볼 수 있다.

## 결과 (프로그램 출력 표를 그대로 붙여넣기)

### A) 블록 크기 스윕 — N = 1e7

| block | grid | total threads | kernel ms | vs CPU | correct |
|---:|---:|---:|---:|---:|:---:|
| | | | | | |

### B) grid-stride — block 256

| grid | total threads | elems/thread | kernel ms | correct |
|---:|---:|---:|---:|:---:|
| | | | | |

## 해석 (직접 적기)
- 32 미만(8, 16)이 느린 이유:
- 100 (32의 배수 아님)은 어땠나, 왜:
- 128 / 256 / 1024 사이 차이가 작은(또는 큰) 이유:
- grid=1 일 때 한 스레드가 몇 개 원소를 맡았고, 시간이 어떻게 됐나:
- 커널 시간 vs H2D/D2H 시간 비교 → 이 예제에서 진짜 병목은:

## 생각해볼 것
- `if (i < n)` 를 빼면 무슨 일이 생기나? (S4 compute-sanitizer 로 다시 확인)
- grid-stride 에서 grid 를 n 보다 크게 주면? (결과는 맞지만 놀고 있는 스레드가 생긴다)
