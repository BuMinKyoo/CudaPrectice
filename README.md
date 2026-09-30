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
├─ S1_ExecModel/
│  ├─ 01_VectorAdd            블록 크기 스윕, grid-stride loop
│  ├─ 02_BgrToGray            2D 인덱싱, BGR → gray, CPU와 바이트 비교
│  └─ 03_AsyncAndErrors       비동기 런치, cudaEvent 시간, 런치 에러
└─ S2_Memory/
   ├─ 04_Transpose            coalescing, 공유 메모리로 접근 순서 바꾸기, bank conflict
   └─ 05_BlurShared           공유 메모리로 중복 읽기 없애기, halo(apron)
```
빌드 결과는 `bin/x64/<Config>/`, 중간 파일은 `obj/` (둘 다 git 제외).

## 시작
1. `CudaPractice.slnx` 열기
2. 구성 **Release | x64**
3. `00_DeviceQuery` 를 시작 프로젝트로 → 실행 → smoke test `OK` 확인
4. 출력된 `sm_XY` 를 `Directory.Build.props` 의 `CudaArch` 에 넣기 (빌드 시간 단축)
5. `01` → `02` → `03` → `04` → `05` 순서로 실행하고 각 README 표 채우기

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
6. `printf` 문자열은 **ASCII 로 끝낸다.** 한글 바로 뒤의 `\n` 은 literal 로 깨진다
   (`-Xcompiler "/utf-8"` 은 cl 에만 가고 nvcc 프론트엔드엔 안 간다)
   - 나쁜 예: `printf("스레드당 %d 칸\n")` → 화면에 `칸\n` 이 그대로 찍힌다
   - 좋은 예: `printf("스레드당 %d cells\n")`

## 진행 상황
| 단계 | 프로젝트 | 상태 |
|---|---|---|
| S0 | 00_DeviceQuery | ⬜ |
| S1 | 01_VectorAdd | ⬜ |
| S1 | 02_BgrToGray | ⬜ |
| S1 | 03_AsyncAndErrors | ⬜ |
| S1 | `preprocess.cu` 읽고 S1 개념 주석 달기 (원본 레포에서) | ⬜ |
| S2 | 04_Transpose | ⬜ |
| S2 | 05_BlurShared | ⬜ |

## 로드맵 (S0~S10)

S0~S2 는 실제로 만들어져 있다. S4~S7 은 기존 예제 주석에 "여기서 다시 본다" 로
예고되어 있어 주제가 정해져 있다. **S3 과 S8~S10 은 아직 정해진 내용이 없다.**

| 단계 | 주제 | 근거 | 필요도 |
|---|---|---|---|
| S0 | 환경 확인, smoke test | 구현됨 | 완료 |
| S1 | 실행 모델 — 블록/warp, 2D 인덱싱, 비동기 런치, 에러 처리 | 구현됨 | 필수 |
| S2 | 메모리 — coalescing, 공유 메모리, bank conflict, halo | 구현됨 | **필수** (성능 대부분이 여기서 나온다) |
| S3 | 미정 | — | — |
| S4 | 검증 — `compute-sanitizer`, sticky 에러, 경쟁 상태 | `01/README.md`, `03/main.cu:20` | **필수** (간헐적 버그는 눈으로 못 찾는다) |
| S5 | 전송 — pinned memory, `cudaMallocPitch` | `01/main.cu`, `02/README.md` | 실시간 파이프라인이면 필수 |
| S6~S7 | 스트림 — CPU·GPU 겹치기, 실시간 루프 | `03/main.cu:78`, `03/README.md` | 실시간 파이프라인이면 필수 |
| S8~S10 | 미정 | — | — |

### 왜 S1 에서 끝내면 안 되는가

02 의 측정값이 답이다.

```
커널 최고 기록 0.122 ms   ←  표에는 "CPU 대비 14.1x" 로 찍힌다
H2D 0.605 ms + D2H 0.309 ms
```

| | 시간 |
|---|---|
| H2D + 커널 + D2H | **1.036 ms** |
| CPU 단독 | 1.720 ms |
| **파이프라인 실질 속도 향상** | **1.66x** |

전체의 **88% 가 복사**다. 커널을 0 으로 만들어도 1.036 → 0.914 ms 가 한계다.
S1 만으로 짠 파이프라인은 개발 복잡도 대비 이득이 거의 없다. 이걸 푸는 도구(pinned, 스트림)가 S5~S7 에 있다.

같은 이야기를 알고리즘 쪽에서 보면 — 04 는 **논리적으로 동일한 전치**인데 방법만 바꿔 3.8배가 난다.

| 04 커널 | 실효 대역폭 |
|---|---|
| TransposeNaive | 30.3 GB/s |
| TransposeShared+pad | **114.0 GB/s** |

즉 S2 이후는 "새로운 문법"이 아니라 **지금 아는 것을 쓸모 있게 만드는 지식**이다.

## 빌드가 안 될 때
| 증상 | 조치 |
|---|---|
| `CUDA 12.9.props` 를 찾을 수 없음 | VS 2026 에 CUDA 통합이 안 깔린 경우. `build/Cuda.props` 가 `%CUDA_PATH_V12_9%\extras\visual_studio_integration\MSBuildExtensions` 로 자동 fallback 한다. 그래도 안 되면 그 폴더 파일 4개를 `C:\Program Files\Microsoft Visual Studio\18\<Edition>\MSBuild\Microsoft\VC\v180\BuildCustomizations\` 에 복사 |
| `MSB8070: MSVC 도구 집합 버전 '14.44.35207' 을 찾을 수 없습니다` | 그 VS 에디션에 14.44 가 없는 것. **VS Installer → 해당 에디션 `수정` → 개별 구성 요소 → `14.44` 검색 → `MSVC v143 - VS 2022 C++ x64/x86 빌드 도구 (v14.44-17.14)` 체크**. 에디션마다 따로 설치해야 한다 (Professional 에 있어도 Community 엔 없을 수 있음) |
| `nvcc error : 'cudafe++' died with status 0xC0000005` | MSVC 14.5x 로 빌드된 것. CUDA 12.x 는 14.4x 까지만 지원한다. `Directory.Build.props` 의 `VCToolsVersion` 이 살아 있는지 확인 |
| `unsupported Microsoft Visual Studio version` | `Directory.Build.props` 의 `CudaAllowUnsupportedCompiler` 가 `true` 인지 확인 |
| STL 에서 `unexpected compiler version` 류 에러 | `build/Common.props` 의 CudaCompile `AdditionalOptions` 에 `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` 추가 |
| C4819 (코드 페이지) 경고, 이상한 문법 에러 | `/utf-8` 이 빠진 것. `build/Common.props` 확인 |
| 실행 중 화면 깜빡 + `cudaErrorLaunchTimeout` | Windows TDR(2초). 커널 작업량을 줄인다 |
| `CUDA driver version is insufficient` | `nvidia-smi` 의 CUDA Version ≥ 12.9 인지 확인, 드라이버 업데이트 |
