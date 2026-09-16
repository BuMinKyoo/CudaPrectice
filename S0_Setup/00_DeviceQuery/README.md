# 00_DeviceQuery — 환경 확인 (S0)

## 목적
빌드 → 링크 → 커널 런치까지 전체 경로가 도는지, 그리고 이후 단계에서 쓸 내 GPU 정보를 한 번에 확인한다.

## 실행 전 체크 (터미널)
```powershell
nvcc --version        # 툴킷 버전 (Directory.Build.props 의 CudaVersion 과 같아야 함)
nvidia-smi            # 드라이버 버전 + 드라이버가 지원하는 CUDA 버전
where compute-sanitizer
```
시작 메뉴에서 **Nsight Compute**, **Nsight Systems** 가 있는지도 본다.

## 내 환경 (채우기)

| 항목 | 값 |
|---|---|
| GPU | |
| Compute Capability | |
| VRAM | |
| 드라이버 / 지원 CUDA | |
| 툴킷(nvcc) | 12.9 |
| Visual Studio | 2026 (v145) |
| SM 개수 | |
| shared memory / block | |
| registers / block | |
| warp size | |
| TDR (kernel timeout) | |
| Nsight Compute / Systems 버전 | |

## 다 했으면
- `Directory.Build.props` 의 `CudaArch` 에 `sm_XY` 를 넣는다 → 빌드가 빨라진다.
- smoke test 가 `OK` 가 아니면 다음 단계로 넘어가지 않는다.
