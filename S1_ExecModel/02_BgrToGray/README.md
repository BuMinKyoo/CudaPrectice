# 02_BgrToGray — 2D 인덱싱 (S1-2)

## 배울 것
- `dim3 block(bx, by)`, `dim3 grid(DivUp(W,bx), DivUp(H,by))` — 이미지 처리의 기본 런치
- `x = blockIdx.x*blockDim.x + threadIdx.x`, `y = ...` 와 경계 검사
- interleaved BGR(HWC) 버퍼의 인덱스 `(y*W + x)*3 + c` — 내 프레임 포맷 그대로
- 블록 모양(정사각 vs 가로로 긴 것)이 시간에 주는 영향
- GPU 결과를 CPU 결과와 **바이트 단위**로 비교하는 습관

## 실행
Release | x64. `02_BgrToGray.exe 3840 2160` 처럼 해상도를 바꿀 수 있다.
실행 후 프로젝트 폴더에 `input_out.ppm`, `gray_out.pgm` 이 생긴다(.gitignore 처리됨).

## 결과 — 1920×1080

| block | threads/block | grid | kernel ms | vs CPU | mismatch |
|---|---:|---|---:|---:|---:|
| | | | | | |

H2D: ___ ms / D2H: ___ ms

## 해석 (직접 적기)
- mismatch 가 0 이 아니었다면 원인:
- 8×8 (64스레드) 과 32×32 (1024스레드) 차이:
- 256×1 처럼 한 줄로 긴 블록은 어땠나. 행 단위로 연속된 주소를 읽는 게 도움이 됐나? (→ S2 coalescing 복선)
- 1080p 한 장 기준 커널 시간 vs 복사 시간 — 실시간 파이프라인에서 어느 쪽을 먼저 줄여야 하나:

## 다음에 연결
- S2: 출력 쓰기가 연속 주소인지, 입력 읽기가 흩어지는지 → coalescing
- S5: `cudaMallocPitch` 로 행 정렬하면 뭐가 달라지나
