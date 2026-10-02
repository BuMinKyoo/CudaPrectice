# 06_Sanitizer — 눈으로 못 잡는 버그를 도구로 잡는다 (S4)

## 왜 지금인가

지금까지 README 들에 미뤄둔 숙제가 쌓여 있다.

| 어디 | 뭐라고 적혀 있었나 |
|---|---|
| `01/README.md` | `if (i < n)` 를 빼면 무슨 일이 생기나? (S4 compute-sanitizer 로 다시 확인) |
| `03/main.cu` | 실행 중 에러(경계 밖 접근)는 sticky → S4 compute-sanitizer |
| `04/README.md` | `__syncthreads()` 를 지우면 **항상** 틀리나, **가끔** 틀리나? |
| `05` | `racy` 커널에서 "실행마다 답이 다른" 현상을 직접 봤다 |

여기서 전부 확인한다.

## 이 예제의 버그들이 가진 공통점

- **컴파일 경고가 없다**
- **크래시도 안 난다** (대부분)
- **테스트를 통과하기도 한다** (가끔)

눈과 단위 테스트로는 못 잡는다. 그래서 도구가 필요하다.

04 에서 "느린 코드도 테스트를 통과한다" 를 봤다면, 여기서는
**"틀린 코드도 테스트를 통과한다"** 를 본다.

## compute-sanitizer 네 가지 도구

| 도구 | 잡는 것 |
|---|---|
| `memcheck` (기본) | 배열 밖 접근, 잘못된 free, 정렬 위반 |
| `racecheck` | **공유 메모리** 경쟁 상태 (전역 메모리는 안 본다) |
| `initcheck` | 초기화 안 된 전역 메모리 읽기 |
| `synccheck` | 잘못된 `__syncthreads()` 사용 |

실행 파일은 CUDA 설치 폴더 안에 있다.

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\compute-sanitizer\compute-sanitizer.exe
```

## 실행

```
06_Sanitizer.exe           사용법 출력
06_Sanitizer.exe 0         정상 동작 (기준)
06_Sanitizer.exe 1         배열 밖 접근
06_Sanitizer.exe 2         공유 메모리 경쟁
06_Sanitizer.exe 3         초기화 안 된 메모리
```

**한 번에 하나씩만 돌린다.** 1번은 CUDA 컨텍스트를 죽이기 때문에 같은 프로세스
안에서 다른 모드를 이어서 돌릴 수 없다. 그것 자체가 1번의 교훈이다.

도구를 걸어 돌리는 법:

```
compute-sanitizer --tool memcheck  06_Sanitizer.exe 1
compute-sanitizer --tool racecheck 06_Sanitizer.exe 2
compute-sanitizer --tool initcheck 06_Sanitizer.exe 3
```

**순서가 중요하다.** 먼저 도구 없이 돌려서 "아무 일도 없어 보이는" 것을 확인하고,
그다음 도구를 걸어 무엇이 잡히는지 비교한다. 그 대비가 이 예제의 전부다.

---

## 모드 1 — 배열 밖 접근

01 의 `if (i < n)` 가드를 뺀 것이다.

```
n = 1000, block = 256  ->  grid = 4  ->  쓰레드 1024개
1000 ~ 1023 번 쓰레드 24명이 할당 범위를 넘어선다
```

이 모드는 **두 부분**으로 나뉜다. 둘의 차이가 요점이다.

### [A] 24칸만 넘어가면 — 에러가 안 난다

```
런치 직후 cudaGetLastError     cudaSuccess
cudaDeviceSynchronize          cudaSuccess        <- 멀쩡하다!
```

`cudaMalloc` 은 요청한 4,000 바이트보다 넉넉하게 잡아준다. 조금 넘친 쓰기는
**할당된 페이지 안**이라 하드웨어가 못 잡는다.

**남의 데이터를 조용히 망가뜨리고 지나간다.** 이것이 최악의 경우다 —
크래시라도 나면 알기라도 하지, 이건 알 방법이 없다.

`memcheck` 를 걸면 24건이 전부 보고된다.

```
========= Invalid __global__ write of size 4 bytes
=========     by thread (232,0,0) in block (3,0,0)
=========     and is 1 bytes after the nearest allocation at 0x... of size 4,000 bytes
```

**쓰레드 번호와 블럭 번호까지 찍어준다.** 도구만이 잡는다.

### [B] 256 MB 너머로 나가면 — 이번엔 잡힌다

```
런치 직후 cudaGetLastError     cudaSuccess             <- 런치는 성공
cudaDeviceSynchronize          cudaErrorIllegalAddress <- 여기서 발견
```

03 에서 본 그대로다. **실행 중 에러는 비동기라 동기화 지점에서야 드러난다.**

### 그리고 sticky

03 에서 "일부러 내지 않는다" 고 미뤄둔 부분이다.

```
cudaMalloc (새 할당)        cudaErrorIllegalAddress   <- 커널과 무관한 호출인데
cudaGetLastError            cudaErrorIllegalAddress
cudaGetLastError (또)       cudaSuccess               <- 슬롯은 비워졌다
cudaMalloc (다시 한 번)     cudaErrorIllegalAddress   <- 그런데 또 실패!
```

**`cudaGetLastError` 를 두 번 부르면 "마지막 에러" 슬롯은 비워진다.
그런데 바로 다음 `cudaMalloc` 은 또 실패한다.** 슬롯만 비워졌을 뿐 컨텍스트는
여전히 죽어 있다. 이것이 sticky 다.

03 의 런치 설정 에러와 비교해 보면 차이가 선명하다.

| | 03 (런치 설정 에러) | 06 (실행 중 에러) |
|---|---|---|
| 예 | `<<<2, 2048>>>` | 배열 밖 접근 |
| 커널이 | **시작조차 안 함** | 실행 중 터짐 |
| `cudaGetLastError` 하면 | 완전히 치워짐 | 슬롯만 비워짐 |
| 이후 API | **정상 동작** | **전부 실패** |
| 복구 | 그냥 계속 | **프로세스 재시작뿐** |

### [A] vs [B] 가 주는 교훈

같은 "배열 밖 쓰기" 인데 **얼마나 멀리 나갔느냐**로 에러 발생 여부가 갈린다.

> **에러가 안 났다고 안전한 것이 아니다.**

---

## 모드 2 — 공유 메모리 경쟁

04 에서 "`__syncthreads()` 를 지우면?" 하고 남겨둔 질문의 실물이다.

```c
__shared__ int s[256];
s[t] = t * 10;
// __syncthreads();          <- 일부러 뺐다
out[t] = s[(t + 1) % blockDim.x];   // 옆 쓰레드가 쓴 칸을 읽는다
```

**내가 쓴 칸이 아니라 옆 쓰레드가 쓴 칸을 읽는다.** 같은 warp 안이면 명령이
같이 진행되므로 우연히 맞는다. warp 경계를 넘는 순간(t=31 이 s[32] 를 읽을 때)
순서가 보장되지 않는다.

### 도구 없이 10번 돌린 결과

```
__syncthreads() 있는 버전:  0 0 0 0 0 0 0 0 0 0
__syncthreads() 없는 버전:  0 0 0 0 0 0 0 0 0 0      <- 전부 통과!
```

**틀린 코드가 10번 다 통과했다.** 에러도 없고 테스트도 통과한다.
이 상태로 몇 년을 쓰다가, GPU 를 바꾸거나 블럭 크기를 조정한 어느 날
갑자기 틀리기 시작한다.

### racecheck 를 걸면

```
=========     and Read access at SharedRace(int *, int)+0x1a8 [1024 hazards]
```

**1024건의 hazard.** 결과가 맞든 틀리든 **"위험한 접근" 자체를 보고한다.**
이것이 단위 테스트와 결정적으로 다른 점이다.

> 테스트는 "이번에 맞았나" 를 묻고, racecheck 는 "틀릴 수 있는 구조인가" 를 묻는다.

`SharedSafe`(동기화 있는 버전)는 보고되지 않는다. 같이 돌려서 대비를 확인할 것.

---

## 모드 3 — 초기화 안 된 메모리

```c
cudaMalloc(&dIn, n * sizeof(int));   // cudaMemset 을 안 했다
SumFrom<<<...>>>(dIn, dOut, n);      // 쓰레기값을 읽는다
```

`cudaMalloc` 은 `malloc` 과 같아서 **0 으로 채워주지 않는다.**

### 도구 없이

```
cudaMalloc 만 하고 cudaMemset 을 안 한 배열의 합: 0
```

**0 이 나왔다. 그런데 이건 우연이다** — 직전에 그 자리를 쓴 사람이 없었을 뿐이다.
프로그램을 더 돌리거나 다른 작업을 섞으면 쓰레기값이 나온다.

### initcheck 를 걸면

```
========= Uninitialized __global__ memory read of size 4 bytes
```

05 에서 `clearOutput()` 으로 매번 `cudaMemset` 한 이유가 이것이다.

---

## 결과 (직접 돌려보고 적기)

| 모드 | 도구 없이 | 도구 걸고 |
|---|---|---|
| 1-A (24칸) | | |
| 1-B (256MB) | | |
| 2 (경쟁) | | |
| 3 (미초기화) | | |

## 해석 (직접 적기)

- 1-A 에서 에러가 안 나는데 1-B 에서는 나는 이유. 둘 다 "배열 밖 쓰기" 인데:
- 1-A 같은 버그가 실제 프로젝트에 있으면 어떻게 드러나게 될까.
  (언제 터질지, 어디서 터진 것처럼 보일지):
- 2 에서 10번 다 통과했다. 그런데 racecheck 는 1024건을 보고했다.
  테스트와 racecheck 가 각각 무엇을 묻고 있는가:
- 3 에서 합이 0 으로 나왔다. 이걸 "초기화가 됐다" 는 증거로 쓸 수 있나:
- sticky 에러를 만났을 때, 에러 메시지만 보고 원인을 찾을 수 있을까.
  (`cudaMalloc` 이 실패했다는 메시지가 어디를 가리키는가):

## 생각해볼 것

- 01 의 `VecAdd` 에서 `if (i < n)` 를 빼고 memcheck 를 걸어보기.
  N 을 256 의 배수로 주면? 아닌 값으로 주면?
- 05 의 `racySumReductionKernel` 에 racecheck 를 걸면 잡히나?
  (힌트: racecheck 는 공유 메모리만 본다. 05 의 경쟁은 어디서 일어나는가?)
- 04 의 `TransposeShared` 에서 `__syncthreads()` 를 지우고 racecheck:
  결과가 틀리나? hazard 는 몇 건인가?
- `--tool synccheck` 로 `__syncthreads()` 를 `if` 안에 넣은 커널을 검사해보기.
  (주의: 교착 상태에 빠져 TDR 2초 리셋이 날 수 있다)
- 도구를 걸면 실행이 느려진다. 몇 배인가? 그래서 언제 쓰는 도구인가:

## 실무에서

```
개발 중        : 새 커널을 짜면 한 번은 memcheck + racecheck 를 통과시킨다
버그가 났을 때  : 답이 이상하다 -> initcheck, 가끔 틀린다 -> racecheck,
                 sticky 에러가 난다 -> memcheck
CI 에서        : 작은 입력으로 memcheck 를 돌린다 (느려서 큰 입력은 무리)
```

**도구를 거는 비용은 느려지는 것뿐이고, 안 걸었을 때의 비용은 며칠이다.**

## 다음에 연결

- **S5**: pinned memory, `cudaMallocPitch`
- **S6~S7**: 스트림. 커널이 겹쳐 돌기 시작하면 경쟁 상태의 범위가 더 넓어진다
  (블럭 간, 커널 간, 스트림 간)
- `cuda-gdb` / Nsight: 여기서 "어디가 틀렸는지" 를 찾았다면,
  그다음은 "왜 그 값이 들어갔는지" 를 보는 단계다
