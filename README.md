# CudaPrectice

CUDA 학습 로드맵(S0~S10)을 **솔루션 1개 + 예제당 프로젝트 1개**로 따라가는 연습 레포.
각 프로젝트는 `main.cu` 하나 + `README.md`(측정값·해석) 하나로 끝난다.

## 환경
| | |
|---|---|
| IDE | Visual Studio 2026 (PlatformToolset v145) |
| CUDA | Toolkit 12.9 — 바꾸려면 `Directory.Build.props` 의 `CudaVersion` 한 줄 |
| 플랫폼 | x64 only (Debug / Release) |
| 솔루션 | `CudaPractice.slnx` |

## 구조
```
CudaPrectice/
├─ CudaPractice.slnx          솔루션 (솔루션 폴더 = 단계)
├─ Directory.Build.props      CUDA 버전, -arch, 공통 루트 경로
├─ build/
│  ├─ Cuda.props / .targets   CUDA 빌드 커스터마이제이션 import (경로 자동 탐색)
│  └─ Common.props            출력 폴더, /utf-8, cudart_static 링크 등 공통 설정
├─ common/cu_common.h         CUDA_CHECK, CUDA_CHECK_LAUNCH, DivUp, GpuTimer, CpuTimer, Median
├─ templates/ProjectTemplate  새 예제 템플릿
├─ tools/new-project.ps1      템플릿 복사 + 솔루션 등록
├─ S0_Setup/
│  └─ 00_DeviceQuery          드라이버/툴킷/GPU 확인 + smoke test
└─ S1_ExecModel/
   ├─ 01_VectorAdd            블록 크기 스윕, grid-stride loop
   ├─ 02_BgrToGray            2D 인덱싱, BGR → gray, CPU와 바이트 비교
   └─ 03_AsyncAndErrors       비동기 런치, cudaEvent 시간, 런치 에러
```
빌드 결과는 `bin/x64/<Config>/`, 중간 파일은 `obj/` (둘 다 git 제외).

## 시작
1. `CudaPractice.slnx` 열기
2. 구성 **Release | x64**
3. `00_DeviceQuery` 를 시작 프로젝트로 → 실행 → smoke test `OK` 확인
4. 출력된 `sm_XY` 를 `Directory.Build.props` 의 `CudaArch` 에 넣기 (빌드 시간 단축)
5. `01` → `02` → `03` 순서로 실행하고 각 README 표 채우기

## 새 예제 추가
```powershell
# 솔루션을 닫고(또는 VS 가 다시 로드하라고 하면 수락)
.\tools\new-project.ps1 -Stage S2_Memory -Name 04_Transpose
```
→ `S2_Memory\04_Transpose\` 에 vcxproj/main.cu/README.md 생성, 솔루션 폴더 `/S2_Memory/` 에 등록.
번호는 단계와 무관하게 전체 일련번호로 이어간다.

## 코드 규칙
1. 모든 CUDA API 호출은 `CUDA_CHECK(...)`
2. 모든 커널 런치 뒤엔 `CUDA_CHECK_LAUNCH()`
3. 시간은 `GpuTimer` + 반복 후 `Median`, 첫 런치는 워밍업으로 버린다
4. GPU 결과는 항상 CPU 기준 구현과 비교한다
5. 숫자는 README 표에, 해석은 한 줄씩

## 진행 상황
| 단계 | 프로젝트 | 상태 |
|---|---|---|
| S0 | 00_DeviceQuery | ⬜ |
| S1 | 01_VectorAdd | ⬜ |
| S1 | 02_BgrToGray | ⬜ |
| S1 | 03_AsyncAndErrors | ⬜ |
| S1 | `preprocess.cu` 읽고 S1 개념 주석 달기 (원본 레포에서) | ⬜ |
| S2 | 04_Transpose, 05_BlurShared, … | 예정 |

## 빌드가 안 될 때
| 증상 | 조치 |
|---|---|
| `CUDA 12.9.props` 를 찾을 수 없음 | VS 2026 에 CUDA 통합이 안 깔린 경우. `build/Cuda.props` 가 `%CUDA_PATH_V12_9%\extras\visual_studio_integration\MSBuildExtensions` 로 자동 fallback 한다. 그래도 안 되면 그 폴더 파일 4개를 `C:\Program Files\Microsoft Visual Studio\18\<Edition>\MSBuild\Microsoft\VC\v180\BuildCustomizations\` 에 복사 |
| `unsupported Microsoft Visual Studio version` | `Directory.Build.props` 의 `CudaAllowUnsupportedCompiler` 가 `true` 인지 확인 |
| STL 에서 `unexpected compiler version` 류 에러 | `build/Common.props` 의 CudaCompile `AdditionalOptions` 에 `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` 추가 |
| C4819 (코드 페이지) 경고, 이상한 문법 에러 | `/utf-8` 이 빠진 것. `build/Common.props` 확인 |
| 실행 중 화면 깜빡 + `cudaErrorLaunchTimeout` | Windows TDR(2초). 커널 작업량을 줄인다 |
| `CUDA driver version is insufficient` | `nvidia-smi` 의 CUDA Version ≥ 12.9 인지 확인, 드라이버 업데이트 |
