# SpacemiT IME Standalone Micro-Kernel Specification

이 문서는 Corenelia MLIR/LLVM 백엔드에서 타겟으로 사용할 외부 어셈블리 커널(`ime_gemm.s`)의 Contract(제약 사항)와 ABI, Semantics를 명시합니다.

## Overview
이 라이브러리는 Banana Pi (SpacemiT K1) 보드의 하드웨어 가속기(IME)를 최대한으로 활용하기 위해 극도로 최적화된 마이크로 커널(Micro-kernel)입니다.
타일 사이즈(4x8x4)가 고정되어 있으며, K 차원 전체의 누산(Accumulation)을 한 번의 함수 호출 내부에서 처리하여 메모리 병목 현상을 방지합니다. 사용자는 내부 어셈블리 구현을 알 필요 없이, 제공되는 C API를 호출하기만 하면 됩니다.

---

## 1. ABI (Application Binary Interface)

입력 행렬(A, B)의 메모리 배치(Layout) 형태에 따라 **세 가지 커널 모드**가 제공되며, 각 모드마다 0으로 초기화하는 버전과 누적(Accumulate)하는 버전이 존재합니다. 모든 커널은 RISC-V 표준 C ABI를 준수합니다.

### A. Pre-packed 모드 (기본 버전)
메모리에 타일 형태(A: 4x8, B: 8x4)로 데이터가 연속 정렬(Packed)되어 있는 경우 사용합니다. 가장 빠르며 `vle8.v` 명령어로 로드합니다.

#### Zero-Initialize Version (`C = A * B`)
```c
void ime_i8_gemm_4x8x4_i8i8i32(
    const int8_t *A_tile_row, // a0: A 타일 시작 주소
    const int8_t *B_tile_col, // a1: B 타일 시작 주소
    int32_t *C_tile,          // a2: C 타일 시작 주소 (결과 덮어쓰기)
    size_t num_K,             // a3: K 타일 루프 개수 (K / 8)
    size_t A_K_stride,        // a4: 다음 K 타일로 이동 시 A 주소 증가량 (bytes)
    size_t B_K_stride         // a5: 다음 K 타일로 이동 시 B 주소 증가량 (bytes)
);
```
- **특징**: 커널 진입 시 내부 누산기를 0으로 초기화하여 완전히 새로운 $C = A \times B$ 연산을 수행합니다. `num_K == 0` 일 경우 메모리 쓰기 없이 안전하게 함수를 종료합니다.

#### Accumulate Version (`C += A * B`)
```c
void ime_i8_gemm_4x8x4_i8i8i32_acc(
    const int8_t *A_tile_row, // a0
    const int8_t *B_tile_col, // a1
    int32_t *C_tile,          // a2 (기존 값 읽어오기 및 덮어쓰기)
    size_t num_K,             // a3
    size_t A_K_stride,        // a4
    size_t B_K_stride         // a5
);
```
- **특징**: 커널 진입 시 기존 C 타일 데이터를 메모리에서 읽어와 누산의 초기값으로 사용합니다. 기존 값에 결과를 더하는 $C \mathrel{+}= A \times B$ 연산에 사용됩니다.

### B. RM-RM 모드 (A: Row-Major, B: Row-Major)
입력 행렬 A와 B가 모두 일반적인 2차원 Row-Major 메모리 구조일 때 사용합니다. `vluxei32.v` (Indexed Load) 명령어를 이용해 내부적으로 타일을 동적 조립하므로, 호출 전에 데이터를 패킹할 필요가 없습니다.

```c
void ime_i8_gemm_4x8x4_i8i8i32_rm_rm(
    const int8_t *A,          // a0: A 행렬의 현재 타일 시작 주소
    const int8_t *B,          // a1: B 행렬의 현재 타일 시작 주소
    int32_t *C_tile,          // a2: C 타일 시작 주소
    size_t num_K,             // a3: K 타일 루프 개수 (K / 8)
    size_t A_K_stride,        // a4: 다음 K 타일로 이동 시 A 주소 증가량 (보통 8)
    size_t B_K_stride,        // a5: 다음 K 타일로 이동 시 B 주소 증가량 (보통 8 * ldb)
    size_t lda,               // a6: A 행렬의 열 개수 (Leading Dimension of A)
    size_t ldb,               // a7: B 행렬의 열 개수 (Leading Dimension of B)
    size_t ldc                // 스택(0(sp)): C 행렬의 열 개수 (Leading Dimension of C)
);

// 누적 버전
void ime_i8_gemm_4x8x4_i8i8i32_rm_rm_acc(...); 
```

### C. RM-CM 모드 (A: Row-Major, B: Column-Major)
행렬 A는 Row-Major, B는 Column-Major 구조일 때 사용합니다. B가 전치(Transposed)되어 들어오는 경우에 유용합니다.

```c
void ime_i8_gemm_4x8x4_i8i8i32_rm_cm(
    const int8_t *A, const int8_t *B, int32_t *C_tile, 
    size_t num_K, size_t A_K_stride, size_t B_K_stride, 
    size_t lda, size_t ldb, size_t ldc
);

// 누적 버전
void ime_i8_gemm_4x8x4_i8i8i32_rm_cm_acc(...);
```

### D. Convolution 3x3 Microkernel (Sliding Window vmadotn)
RISC-V Zvvm (SpacemiT AI) 확장의 핵심인 `vmadotN` 명령어(Sliding Window)를 활용하여 2D Convolution 연산을 최적화한 마이크로커널입니다. 3x3 가중치 커널을 사용하여 한 번에 4x4 크기의 출력 타일(Output Width 4픽셀 × Filter 4채널)을 계산합니다.

```c
void ime_i8_conv2d_3x3_4x4_tile_i8i8i32_acc(
    const int8_t *Act,        // a0: [N, C_in/8, H, W, 8]
    const int8_t *Weight,     // a1: [F/4, C_in/8, KH, KW, 4, 8]
    int32_t *Out,             // a2: [N, F/4, Ho, Wo, 4]
    size_t num_C,             // a3: C_in / 8 차원 루프 횟수
    size_t act_C_stride,      // a4: 다음 C_in 블록 이동 시 Act 주소 증가량 (bytes)
    size_t weight_C_stride,   // a5: 다음 C_in 블록 이동 시 Weight 주소 증가량 (bytes)
    size_t act_H_stride,      // a6: 다음 H 줄로 이동 시 Act 주소 증가량 (bytes)
    size_t act_W_stride       // a7: (미사용, 향후 확장용)
);
```
- **특징**: 
  - 입력 Activation(Act)을 단 한 번만 로드(`vle8.v`, `LMUL=2`)하여 64바이트(8 픽셀)를 벡터 레지스터에 넉넉히 올립니다.
  - 이후 `vmadot`, `vmadot1`, `vmadot2` 등 명령어 레벨의 시프트(오프셋) 연산을 사용하여, **메모리 재접근이나 추가적인 레지스터 이동(vslide 등) 없이 하드웨어적으로 슬라이딩 윈도우 내적 연산을 수행**합니다.
  - 이로 인해 Unaligned load나 Im2Col 오버헤드를 원천 차단하여 극도의 Compute-Bound 성능을 달성합니다.

### E. Calling Convention (호출 규약)
- 본 바이너리는 **RISC-V 표준 C ABI**를 완벽하게 준수합니다.
- 함수 내부에서 Caller-saved 레지스터(스칼라 및 벡터 레지스터 포함)를 사용할 수 있으므로, 컴파일러는 ABI 규약에 따라 레지스터를 보존(Save/Restore)해야 합니다.
- 스택(Stack) 메모리를 전혀 사용하지 않는 Leaf function으로 구현되어 있어 함수 호출 오버헤드가 극히 낮습니다.

---

## 2. Layout Constraints (모드별 제약 사항)

Corenelia 백엔드는 `linalg.matmul`을 이 커널로 lowering 할 때 **사용하려는 커널 모드에 맞춰 메모리 레이아웃 조건을 보장**해야 합니다.

### A. Matrix A (Int8)
- **Shape**: $4 \times 8$ (Ti=4, Tk=8)
- **Layout**: 
  - **Pre-packed 모드**: 32바이트 타일 전체가 메모리 상에 완전히 연속적(Contiguous)이어야 합니다.
  - **RM-RM / RM-CM 모드**: 행렬의 Leading Dimension인 `lda` 값을 전달해주면 커널 내부에서 `vluxei32.v`를 사용해 알아서 연속된 4개의 8원소 벡터로 조립하므로, 타일 전체 메모리가 물리적으로 연속적일 필요는 없습니다. (각 행의 8개 원소는 메모리 상에 연속이어야 함)

### B. Matrix B (Int8)
- **Shape**: $8 \times 4$ (Tk=8, Tj=4)
- **Layout**: 
  - **Pre-packed 모드**: **IME Packed Layout**. 즉, 논리적인 행/열 구조와 관계없이 32바이트가 메모리 상에 연속적으로 배치되어야 하며, 하드웨어 가속기(IME)가 기대하는 내부 포맷으로 런타임 전에 미리 정렬(Transposed)되어 있어야 합니다.
  - **RM-RM / RM-CM 모드**: 원래 배열이 Row-Major이든 Col-Major이든 `ldb` 파라미터만 넘겨주면, 커널이 Indexed Load 오프셋 계산을 통해 자동으로 하드웨어가 요구하는 **IME Packed Layout** 구조로 벡터 레지스터에 동적 적재(Transpose)합니다. 사전에 패킹 작업을 할 필요가 없습니다.

### C. Matrix C (Int32)
- **Shape**: $4 \times 4$ (Ti=4, Tj=4)
- **Layout**: 
  - **Pre-packed 모드**: **반드시 연속된 64바이트(Contiguous Physical Layout)**여야 합니다. 이 모드는 최고의 하드웨어 한계 성능 측정을 목표로 하므로, `vse32.v`를 사용해 단일 사이클로 결과를 쏟아냅니다. C가 큰 $M \times N$ 행렬의 부분 뷰일 경우 임시 버퍼 포인터를 넘겨야 합니다.
  - **RM-RM / RM-CM 모드**: 이 커널들은 Packed 레이아웃을 쓰지 않았을 때의 진정한 "메모리 접근 오버헤드"를 벤치마킹하기 위해 존재합니다. 따라서 C 타일 역시 `ldc` 값을 넘겨주면 커널 내부에서 `vsoxei32.v` (Indexed Store)를 이용해 런타임에 직접 $M \times N$ 행렬의 원래 위치로 분산 저장(Scatter)합니다. (오버헤드가 정확히 포함됩니다)

### D. Convolution Input/Weight/Output Layout
Convolution 커널(`ime_i8_conv2d_3x3_4x4_tile_i8i8i32_acc`)에 적용되는 레이아웃 제약 사항입니다.
- **Act (입력 픽셀)**: `[N, C_in/8, H, W, 8]` 구조. C_in 채널이 8바이트 단위로 패킹되어 있으며, 메모리 상에 가로(W) 픽셀들이 8바이트(채널 패킹) 단위로 연속적으로 배치되어야 `vmadot1` 등의 슬라이딩 윈도우 연산이 정확히 1 픽셀씩 시프트하며 올바르게 작동합니다.
- **Weight (가중치)**: `[F/4, C_in/8, KH, KW, 4, 8]` 구조. 4개의 필터 타일이 32바이트(4 필터 x 8 입력 채널) 단위로 연속적으로 저장되어 있어야 합니다.
- **Output (출력)**: `[N, F/4, Ho, Wo, 4]` 구조. $4 \times 4$ 타일(64바이트)이 물리적으로 연속되어야 단일 쓰기 명령(`vse32.v`)으로 처리 가능합니다.

---

## 3. Hardware Assumptions

- **VLEN = 256 bits**: 이 바이너리 라이브러리는 하드웨어의 벡터 레지스터 길이(VLEN)가 256비트 환경임을 전제로 최적화되어 있습니다. VLEN이 다른 일반 RISC-V 보드에서는 정상적으로 동작하지 않습니다.
- **SpacemiT IME 지원**: Banana Pi K1 코어에 탑재된 SpacemiT 벤더 특화 매트릭스 확장(IME)을 활용합니다. 따라서 해당 확장이 지원되지 않는 에뮬레이터(Spike 등)나 보드에서는 실행(Illegal Instruction)이 불가능합니다.
