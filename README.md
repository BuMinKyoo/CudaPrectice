# CudaPrectice

CUDA 기본기를 **솔루션 1개 + 예제당 프로젝트 1개**로 따라가는 연습 레포.
각 프로젝트는 `main.cu` 하나 + `README.md`(측정값·해석) 하나로 끝난다.

**S0 ~ S6 으로 기본기는 한 바퀴 돌았다.**

```
S0 환경  →  S1 실행 모델  →  S2 메모리  →  S3 리덕션
                                              ↓
         S6 라이브러리  ←  S5 스트림  ←  S4 검증
```

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
├─ S2_Memory/
│  └─ 04_Transpose            coalescing, 공유 메모리, bank conflict
├─ S3_Reduction/
│  └─ 05_Reduction            경쟁 상태, atomicAdd 경합, 트리 접기, float 결합법칙
├─ S4_Debugging/
│  └─ 06_Sanitizer            compute-sanitizer 로 memcheck / racecheck / initcheck
├─ S5_Streams/
│  └─ 07_Streams              pinned memory, cudaMemcpyAsync, 복사와 커널 겹치기
└─ S6_Libraries/
   └─ 08_Thrust               직접 짠 것 vs thrust 한 줄. 언제 직접 짜나
```
빌드 결과는 `bin/x64/<Config>/`, 중간 파일은 `obj/` (둘 다 git 제외).

## 시작
1. `CudaPractice.slnx` 열기
2. 구성 **Release | x64**
3. `00_DeviceQuery` 를 시작 프로젝트로 → 실행 → smoke test `OK` 확인
4. 출력된 `sm_XY` 를 `Directory.Build.props` 의 `CudaArch` 에 넣기 (빌드 시간 단축)
5. `01` 부터 `08` 까지 번호 순서로 실행하고 각 README 표 채우기

순서가 중요하다. 뒤 예제가 앞 예제의 결과를 근거로 쓴다.
예를 들어 `08_Thrust` 는 `05_Reduction` 을 직접 짜 본 다음에야 의미가 있다.

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
6. `printf` 문자열은 **ASCII 로 끝낸다.** 한글 바로 뒤에 `\` 를 쓰면 삼켜질 수 있다
   (`-Xcompiler "/utf-8"` 은 cl 에만 가고 nvcc 프론트엔드엔 안 간다)
   - 나쁜 예: `printf("스레드당 %d 칸\n")` → 화면에 `칸\n` 이 그대로 찍힌다
   - 좋은 예: `printf("스레드당 %d cells\n")`
   - `\"` 는 더 심하다. 닫는 따옴표가 먹혀서 **컴파일이 깨진다** (06 에서 겪음)
   - 주의: **항상 깨지는 것은 아니다.** `시간\n` 은 멀쩡한데 `칸\n` 은 깨진다.
     어떤 글자에서 터지는지 규칙을 특정하지 못했으므로, 그냥 ASCII 로 끝내는 것이 답이다

## 진행 상황
| 단계 | 프로젝트 | 상태 |
|---|---|---|
| S0 | 00_DeviceQuery | ⬜ |
| S1 | 01_VectorAdd | ⬜ |
| S1 | 02_BgrToGray | ⬜ |
| S1 | 03_AsyncAndErrors | ⬜ |
| S1 | `preprocess.cu` 읽고 S1 개념 주석 달기 (원본 레포에서) | ⬜ |
| S2 | 04_Transpose | ⬜ |
| S3 | 05_Reduction | ⬜ |
| S4 | 06_Sanitizer | ⬜ |
| S5 | 07_Streams | ⬜ |
| S6 | 08_Thrust | ⬜ |

## 로드맵 — 무엇을 왜 했나

| 단계 | 질문 | 답 |
|---|---|---|
| **S0** | 내 GPU 는 무엇인가 | SM 수, warp 32, 블럭/SM 한도 — 이후 모든 계산의 기준값 |
| **S1** | GPU 에서 코드가 돌게 하려면 | 블럭/warp/스레드, 2D 인덱싱, 비동기 런치, 에러 처리 |
| **S2** | 왜 느린가 | **메모리 접근 패턴.** 같은 알고리즘이 3.8배 차이 난다 |
| **S3** | 스레드들이 협력하려면 | 경쟁 상태 → atomic → 트리로 접기 |
| **S4** | 맞는지 어떻게 아는가 | 눈과 테스트로는 못 잡는다. `compute-sanitizer` |
| **S5** | 복사 비용을 줄이려면 | pinned + 비동기 + 스트림으로 겹치기 |
| **S6** | 직접 짜야 하는가 | 이름이 붙은 연산은 라이브러리. **thrust 가 4배 빠르고 7배 정확하다** |

### 각 단계가 앞 단계의 한계를 푼다

```
S1  블럭 크기만 맞추면 되는 줄 알았다
      -> 02 에서 같은 스레드 수인데 블럭 모양만 바꿔 2.1배 차이
S2  접근 패턴이 문제였다. 04 에서 3.8배
      -> 그런데 출력이 하나인 문제(합계)는 이 방법으로 안 된다
S3  트리로 접어서 해결. 그런데 __syncthreads() 를 빠뜨리면?
      -> 테스트가 통과해 버린다 (06 에서 10번 다 통과)
S4  도구로 잡는다. 이제 커널은 맞다
      -> 그런데 01 에서 본 "전체의 88%가 복사" 는 그대로다
S5  겹쳐서 숨긴다
      -> 여기까지 하고 나니, 05 를 직접 짤 필요가 있었나?
S6  없다. 하지만 05 를 해봤기 때문에 왜 thrust 가 빠른지 안다
```

### 숫자로 보는 요약 (Quadro P2000)

| 어디 | 무엇 | 효과 |
|---|---|---|
| 02 | 블럭 모양 8x8 → 64x4 | **2.1x** |
| 04 | Naive → Shared+pad (같은 전치) | **3.8x** |
| 05 | atomic → segmented | **24.5x** + 정확도 100배 |
| 07 | 동기 → pinned + 스트림 | 1.1x (환경 한계, README 참고) |
| 08 | 05 직접 → `thrust::reduce` | **4x** + 정확도 7배 |

**전부 "같은 일을 하는 다른 방법" 의 차이다.** 새 문법이 아니라
아는 것을 쓸모 있게 만드는 지식이었다.

### 여기까지 하면

남의 CUDA 코드를 읽고, 이해하고, 고칠 수 있다. **기본기는 여기까지다.**

다음은 실제로 만들고 싶은 것을 만들면서 필요한 것만 그때그때 배우면 된다.
남은 것들(scan, warp shuffle, cooperative groups, multi-GPU, 텐서코어,
`cudaMallocPitch`)은 **필요해지면 그때 배우는 것**이지 미리 해둘 것은 아니다.

### 아직 안 한 것

- 각 예제 README 의 **결과표·해석란 채우기** — 본인 GPU 로 돌려보고 직접 적는 부분
- `ncu` / `nsys` 로 복습 — 04 의 bank conflict, 05 의 occupancy, 07 의 겹침이
  실제로 일어났는지 **추측이 아니라 숫자로** 확인하기.
  각 예제 README 의 "확인해볼 도구" 에 명령어가 적혀 있다
- `preprocess.cu` 읽고 S1~S3 개념 주석 달기 (원본 레포에서)

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
