# SpacemiT IME Standalone Micro-Kernel Specification

이 문서는 Corenelia MLIR/LLVM 백엔드에서 타겟으로 사용할 외부 어셈블리 커널(`ime_gemm.s`)의 Contract(제약 사항)와 ABI, Semantics를 명시합니다.

## Overview
이 라이브러리는 Banana Pi (SpacemiT K1) 보드의 하드웨어 가속기(IME)를 최대한으로 활용하기 위해 극도로 최적화된 마이크로 커널(Micro-kernel)입니다.
타일 사이즈(4x8x4)가 고정되어 있으며, K 차원 전체의 누산(Accumulation)을 한 번의 함수 호출 내부에서 처리하여 메모리 병목 현상을 방지합니다. 사용자는 내부 어셈블리 구현을 알 필요 없이, 제공되는 C API를 호출하기만 하면 됩니다.

---

## 1. ABI (Application Binary Interface)

두 종류의 커널이 제공되며 RISC-V 표준 C ABI를 준수합니다.

### A. Zero-Initialize Version (`C = A * B`)
```c
void ime_i8_gemm_4x8x4_i8i8i32(
    const int8_t *A_tile_row, // a0: A 타일 시작 주소
    const int8_t *B_tile_col, // a1: B 타일 시작 주소
    int32_t *C_tile,          // a2: C 타일 시작 주소 (결과 덮어쓰기)
    size_t num_K,             // a3: K 타일 루프 개수 (K / 8)
    size_t A_stride,          // a4: 다음 K 타일로 이동 시 A 주소 증가량 (bytes)
    size_t B_stride           // a5: 다음 K 타일로 이동 시 B 주소 증가량 (bytes)
);
```
- **특징**: 커널 진입 시 내부 누산기를 0으로 초기화하여 완전히 새로운 $C = A \times B$ 연산을 수행합니다. `num_K == 0` 일 경우 메모리 쓰기 없이 안전하게 함수를 종료합니다.

### B. Accumulate Version (`C += A * B`)
```c
void ime_i8_gemm_4x8x4_i8i8i32_acc(
    const int8_t *A_tile_row, // a0
    const int8_t *B_tile_col, // a1
    int32_t *C_tile,          // a2 (기존 값 읽어오기 및 덮어쓰기)
    size_t num_K,             // a3
    size_t A_stride,          // a4
    size_t B_stride           // a5
);
```
- **특징**: 커널 진입 시 기존 C 타일 데이터를 메모리에서 읽어와 누산의 초기값으로 사용합니다. 기존 값에 결과를 더하는 $C \mathrel{+}= A \times B$ 연산에 사용됩니다.

### C. Calling Convention (호출 규약)
- 본 바이너리는 **RISC-V 표준 C ABI**를 완벽하게 준수합니다.
- 함수 내부에서 Caller-saved 레지스터(스칼라 및 벡터 레지스터 포함)를 사용할 수 있으므로, 컴파일러는 ABI 규약에 따라 레지스터를 보존(Save/Restore)해야 합니다.
- 스택(Stack) 메모리를 전혀 사용하지 않는 Leaf function으로 구현되어 있어 함수 호출 오버헤드가 극히 낮습니다.

---

## 2. Layout Constraints (가장 중요)

Corenelia 백엔드는 `linalg.matmul`을 이 커널로 lowering 할 때 **반드시 아래의 메모리 레이아웃 조건을 보장**해야 합니다.

### A. Matrix A (Int8)
- **Shape**: $4 \times 8$ (Ti=4, Tk=8)
- **Layout**: 32바이트가 메모리 상에 완전히 연속적(Contiguous)이어야 합니다.
- `A_stride`는 일반적으로 32를 넘겨주게 되며, Corenelia 측의 메모리 포맷에 따라 유연하게 제어 가능합니다.

### B. Matrix B (Int8)
- **Shape**: $8 \times 4$ (Tk=8, Tj=4)
- **Layout**: **IME Packed Layout**. 즉, 논리적인 행/열 구조와 관계없이 32바이트가 메모리 상에 연속적으로 배치되어야 하며, 하드웨어 가속기(IME)가 기대하는 내부 포맷(Virtual Dimension Transpose 적용)으로 정렬되어 있어야 합니다.

### C. Matrix C (Int32)
- **Shape**: $4 \times 4$ (Ti=4, Tj=4)
- **Layout**: **반드시 연속된 64바이트(Contiguous Physical Layout)**여야 합니다.
- **이유**: 최상의 성능을 내기 위해 라이브러리 내부에서 단일 벡터 메모리 명령어를 사용하여 16개의 int32 값을 한 번에 기록합니다. 만약 C가 큰 $M \times N$ 행렬의 부분 뷰(Sub-view) 형태라면 Stride가 발생하므로 이 커널에 포인터를 직접 넘겨줄 수 없습니다. 
- **해결책**: Corenelia 컴파일러는 타일 루프를 구성할 때 C 텐서 역시 Virtual Dimension 구조로 미리 패킹해두거나, Scratch Buffer(64바이트) 포인터를 커널에 넘겨 결과를 받은 뒤 Scatter 방식으로 본래 메모리에 쓰는 별도의 로직을 추가해야 합니다.

---

## 3. Hardware Assumptions

- **VLEN = 256 bits**: 이 바이너리 라이브러리는 하드웨어의 벡터 레지스터 길이(VLEN)가 256비트 환경임을 전제로 최적화되어 있습니다. VLEN이 다른 일반 RISC-V 보드에서는 정상적으로 동작하지 않습니다.
- **SpacemiT IME 지원**: Banana Pi K1 코어에 탑재된 SpacemiT 벤더 특화 매트릭스 확장(IME)을 활용합니다. 따라서 해당 확장이 지원되지 않는 에뮬레이터(Spike 등)나 보드에서는 실행(Illegal Instruction)이 불가능합니다.
