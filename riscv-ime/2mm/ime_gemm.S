    .text
    .global ime_i8_gemm_4x8x4_i8i8i32
    .type ime_i8_gemm_4x8x4_i8i8i32, @function
    .global ime_i8_gemm_4x8x4_i8i8i32_acc
    .type ime_i8_gemm_4x8x4_i8i8i32_acc, @function
    .align 2

// C ABI Signature:
// void ime_i8_gemm_4x8x4_i8i8i32(
//     const int8_t *A_tile_row, // a0
//     const int8_t *B_tile_col, // a1
//     int32_t *C_tile,          // a2
//     size_t num_K,             // a3
//     size_t A_stride,          // a4
//     size_t B_stride           // a5
// );
// Semantics: C = A * B (Overwrites C)
// Contract: Requires VLEN=256-bit. C must be a contiguous 4x4 physical layout.

ime_i8_gemm_4x8x4_i8i8i32:
    beqz        a3, .L_return        // num_K == 0 이면 메모리를 덮어쓰지 않고 즉시 반환

    // 누산기(v28)를 0으로 초기화
    // C 타일은 16개의 int32_t = 64바이트. VLEN=256 bits(32 bytes)이므로 LMUL=2 (m2) 설정
    vsetvli     t0, zero, e32, m2
    vxor.vv     v28, v28, v28

    j           .L_loop_setup

// C ABI Signature:
// void ime_i8_gemm_4x8x4_i8i8i32_acc(
//     const int8_t *A_tile_row, // a0
//     const int8_t *B_tile_col, // a1
//     int32_t *C_tile,          // a2
//     size_t num_K,             // a3
//     size_t A_stride,          // a4
//     size_t B_stride           // a5
// );
// Semantics: C += A * B (Accumulates into existing C)
// Contract: Requires VLEN=256-bit. C must be a contiguous 4x4 physical layout.

ime_i8_gemm_4x8x4_i8i8i32_acc:
    beqz        a3, .L_return        // num_K == 0 이면 즉시 반환

    // 기존 C 타일을 메모리에서 로드하여 누산기에 저장
    vsetvli     t0, zero, e32, m2
    vle32.v     v28, (a2)

.L_loop_setup:
    // K 타일 루프 환경 설정
    // A, B 타일은 각각 32바이트(Ti=4, Tk=8 / Tk=8, Tj=4)이므로 LMUL=1 (m1) 설정
    vsetvli     t0, zero, e8, m1

.L_loop_k:
    // A 타일 로드 (32 bytes)
    vle8.v      v0, (a0)
    // B 타일 로드 (32 bytes)
    vle8.v      v1, (a1)

    // vmadot 누산 연산 (v28 += v0 * v1)
    vmadot      v28, v0, v1

    // 포인터 및 카운터 갱신
    add         a0, a0, a4           // A_tile_row += A_stride
    add         a1, a1, a5           // B_tile_col += B_stride
    addi        a3, a3, -1           // num_K--

    bnez        a3, .L_loop_k        // if num_K != 0, loop

    // 최종 결과 저장
    // C 타일을 저장하기 위해 다시 e32, m2 설정
    vsetvli     t0, zero, e32, m2
    vse32.v     v28, (a2)

.L_return:
    ret
