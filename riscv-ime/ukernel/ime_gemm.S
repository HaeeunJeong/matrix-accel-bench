    .text
    .global ime_i8_gemm_4x8x4_i8i8i32
    .type ime_i8_gemm_4x8x4_i8i8i32, @function
    .global ime_i8_gemm_4x8x4_i8i8i32_acc
    .type ime_i8_gemm_4x8x4_i8i8i32_acc, @function
    .align 2

// C ABI Signature:
// void ime_i8_gemm_4x8x4_i8ii32(
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

// ============================================================================
// RM-RM (Row-Major A, Row-Major B) Mode
// ============================================================================
// C ABI Signature:
// void ime_i8_gemm_4x8x4_i8i8i32_rm_rm(
//     const int8_t *A,          // a0
//     const int8_t *B,          // a1
//     int32_t *C,               // a2
//     size_t num_K,             // a3
//     size_t A_K_stride,        // a4
//     size_t B_K_stride,        // a5
//     size_t lda,               // a6
//     size_t ldb,               // a7
//     size_t ldc                // 0(sp)
// );

    .global ime_i8_gemm_4x8x4_i8i8i32_rm_rm
    .type ime_i8_gemm_4x8x4_i8i8i32_rm_rm, @function
    .global ime_i8_gemm_4x8x4_i8i8i32_rm_rm_acc
    .type ime_i8_gemm_4x8x4_i8i8i32_rm_rm_acc, @function
    .align 2

ime_i8_gemm_4x8x4_i8i8i32_rm_rm:
    beqz        a3, .L_rm_rm_return

    // C Index Vector (v24) 계산
    ld          t3, 0(sp)          // t3 = ldc
    slli        t3, t3, 2          // t3 = ldc * 4 (bytes per row)
    vsetvli     t0, zero, e32, m2
    vid.v       v24
    vsrl.vi     v20, v24, 2        // v20 = row = vid / 4
    vand.vi     v22, v24, 3        // v22 = col = vid % 4
    vmul.vx     v26, v20, t3       // v26 = row * (ldc * 4)
    vsll.vi     v22, v22, 2        // v22 = col * 4
    vadd.vv     v24, v26, v22      // v24 = byte offset for C

    // C 타일 누산기(v28) 0으로 초기화
    vxor.vv     v28, v28, v28
    j           .L_rm_rm_setup_ab

ime_i8_gemm_4x8x4_i8i8i32_rm_rm_acc:
    beqz        a3, .L_rm_rm_return

    // C Index Vector (v24) 계산
    ld          t3, 0(sp)          // t3 = ldc
    slli        t3, t3, 2          // t3 = ldc * 4 (bytes per row)
    vsetvli     t0, zero, e32, m2
    vid.v       v24
    vsrl.vi     v20, v24, 2
    vand.vi     v22, v24, 3
    vmul.vx     v26, v20, t3
    vsll.vi     v22, v22, 2
    vadd.vv     v24, v26, v22

    // 기존 C 타일 로드 (Indexed Load)
    vluxei32.v  v28, (a2), v24

.L_rm_rm_setup_ab:
    // A Index Vector (v4): offset = (vid / 8) * lda + (vid % 8)
    vsetvli     t0, zero, e32, m4
    vid.v       v4
    vsrl.vi     v12, v4, 3
    vand.vi     v16, v4, 7
    vmul.vx     v20, v12, a6
    vadd.vv     v4, v20, v16

    // B Index Vector (v8) RM 모드: offset = (vid % 8) * ldb + (vid / 8)
    vid.v       v8
    vmul.vx     v20, v16, a7
    vadd.vv     v8, v20, v12

    // K 타일 루프 환경 설정 (e8 data, m1 data, m4 index)
    vsetvli     t0, zero, e8, m1

.L_rm_rm_loop:
    // A 타일, B 타일 Indexed Load
    vluxei32.v  v0, (a0), v4
    vluxei32.v  v1, (a1), v8

    // vmadot 누산 연산
    vmadot      v28, v0, v1

    // 포인터 및 카운터 갱신
    add         a0, a0, a4
    add         a1, a1, a5
    addi        a3, a3, -1

    bnez        a3, .L_rm_rm_loop

    // 최종 결과 저장 (Indexed Store)
    vsetvli     t0, zero, e32, m2
    vsoxei32.v  v28, (a2), v24

.L_rm_rm_return:
    ret

// ============================================================================
// RM-CM (Row-Major A, Col-Major B) Mode
// ============================================================================
// C ABI Signature:
// void ime_i8_gemm_4x8x4_i8i8i32_rm_cm(
//     const int8_t *A,          // a0
//     const int8_t *B,          // a1
//     int32_t *C,               // a2
//     size_t num_K,             // a3
//     size_t A_K_stride,        // a4
//     size_t B_K_stride,        // a5
//     size_t lda,               // a6
//     size_t ldb,               // a7
//     size_t ldc                // 0(sp)
// );

    .global ime_i8_gemm_4x8x4_i8i8i32_rm_cm
    .type ime_i8_gemm_4x8x4_i8i8i32_rm_cm, @function
    .global ime_i8_gemm_4x8x4_i8i8i32_rm_cm_acc
    .type ime_i8_gemm_4x8x4_i8i8i32_rm_cm_acc, @function
    .align 2

ime_i8_gemm_4x8x4_i8i8i32_rm_cm:
    beqz        a3, .L_rm_cm_return

    // C Index Vector (v24) 계산
    ld          t3, 0(sp)          // t3 = ldc
    slli        t3, t3, 2          // t3 = ldc * 4 (bytes per row)
    vsetvli     t0, zero, e32, m2
    vid.v       v24
    vsrl.vi     v20, v24, 2        // v20 = row = vid / 4
    vand.vi     v22, v24, 3        // v22 = col = vid % 4
    vmul.vx     v26, v20, t3       // v26 = row * (ldc * 4)
    vsll.vi     v22, v22, 2        // v22 = col * 4
    vadd.vv     v24, v26, v22      // v24 = byte offset for C

    // C 타일 누산기 0으로 초기화
    vxor.vv     v28, v28, v28
    j           .L_rm_cm_setup_ab

ime_i8_gemm_4x8x4_i8i8i32_rm_cm_acc:
    beqz        a3, .L_rm_cm_return

    // C Index Vector (v24) 계산
    ld          t3, 0(sp)          // t3 = ldc
    slli        t3, t3, 2          // t3 = ldc * 4 (bytes per row)
    vsetvli     t0, zero, e32, m2
    vid.v       v24
    vsrl.vi     v20, v24, 2
    vand.vi     v22, v24, 3
    vmul.vx     v26, v20, t3
    vsll.vi     v22, v22, 2
    vadd.vv     v24, v26, v22

    // 기존 C 타일 로드 (Indexed Load)
    vluxei32.v  v28, (a2), v24

.L_rm_cm_setup_ab:
    // A Index Vector (v4): offset = (vid / 8) * lda + (vid % 8)
    vsetvli     t0, zero, e32, m4
    vid.v       v4
    vsrl.vi     v12, v4, 3
    vand.vi     v16, v4, 7
    vmul.vx     v20, v12, a6
    vadd.vv     v4, v20, v16

    // B Index Vector (v8) CM 모드: offset = (vid / 8) * ldb + (vid % 8)
    // A와 인덱스 로직이 동일 (stride만 ldb 사용)
    vid.v       v8
    vmul.vx     v20, v12, a7
    vadd.vv     v8, v20, v16

    vsetvli     t0, zero, e8, m1

.L_rm_cm_loop:
    vluxei32.v  v0, (a0), v4
    vluxei32.v  v1, (a1), v8

    vmadot      v28, v0, v1

    add         a0, a0, a4
    add         a1, a1, a5
    addi        a3, a3, -1

    bnez        a3, .L_rm_cm_loop

    // 최종 결과 저장 (Indexed Store)
    vsetvli     t0, zero, e32, m2
    vsoxei32.v  v28, (a2), v24

.L_rm_cm_return:
    ret

// ============================================================================
// Convolution 3x3 Microkernel (Sliding Window vmadotn)
// ============================================================================
// C ABI Signature:
// void ime_i8_conv2d_3x3_4x4_tile_i8i8i32_acc(
//     const int8_t *Act,        // a0: [N, C_in/8, H, W, 8]
//     const int8_t *Weight,     // a1: [F/4, C_in/8, KH, KW, 4, 8]
//     int32_t *Out,             // a2: [N, F/4, Ho, Wo, 4]
//     size_t num_C,             // a3: C_in / 8
//     size_t act_C_stride,      // a4
//     size_t weight_C_stride,   // a5
//     size_t act_H_stride,      // a6
//     size_t act_W_stride       // a7
// );
// Computes a 4x4 output tile (4 spatial pixels along W, 4 filters along F)
// using vmadot, vmadot1, vmadot2.

    .global ime_i8_conv2d_3x3_4x4_tile_i8i8i32_acc
    .type ime_i8_conv2d_3x3_4x4_tile_i8i8i32_acc, @function
    .align 2

ime_i8_conv2d_3x3_4x4_tile_i8i8i32_acc:
    beqz        a3, .L_conv_return

    // Load existing Out (4x4 tile) into v28
    vsetvli     t0, zero, e32, m2
    vle32.v     v28, (a2)

.L_conv_c_loop:
    // C loop setup
    mv          t1, a0       // t1 = Act_ptr for this c_in block
    mv          t2, a1       // t2 = Weight_ptr for this c_in block
    
    // Kh = 0
    // Kw = 0, 1, 2
    // We load Act for H=0. We need W=0..5 (6 pixels).
    // vle8.v with m2 loads 64 bytes (8 pixels = W=0..7). This is enough for kw=0,1,2.
    vsetvli     t0, zero, e8, m2
    vle8.v      v0, (t1)            // v0, v1 = Act[H=0, W=0..7]
    
    vsetvli     t0, zero, e8, m1
    vle8.v      v4, (t2)            // v4 = Weight[KH=0, KW=0]
    vmadot      v28, v0, v4
    addi        t2, t2, 32
    
    vle8.v      v5, (t2)            // v5 = Weight[KH=0, KW=1]
    vmadot1     v28, v0, v5
    addi        t2, t2, 32
    
    vle8.v      v6, (t2)            // v6 = Weight[KH=0, KW=2]
    vmadot2     v28, v0, v6
    addi        t2, t2, 32
    
    // Kh = 1
    add         t1, t1, a6          // t1 += act_H_stride
    vsetvli     t0, zero, e8, m2
    vle8.v      v0, (t1)            // v0, v1 = Act[H=1, W=0..7]
    
    vsetvli     t0, zero, e8, m1
    vle8.v      v4, (t2)            // v4 = Weight[KH=1, KW=0]
    vmadot      v28, v0, v4
    addi        t2, t2, 32
    
    vle8.v      v5, (t2)            // v5 = Weight[KH=1, KW=1]
    vmadot1     v28, v0, v5
    addi        t2, t2, 32
    
    vle8.v      v6, (t2)            // v6 = Weight[KH=1, KW=2]
    vmadot2     v28, v0, v6
    addi        t2, t2, 32
    
    // Kh = 2
    add         t1, t1, a6          // t1 += act_H_stride
    vsetvli     t0, zero, e8, m2
    vle8.v      v0, (t1)            // v0, v1 = Act[H=2, W=0..7]
    
    vsetvli     t0, zero, e8, m1
    vle8.v      v4, (t2)            // v4 = Weight[KH=2, KW=0]
    vmadot      v28, v0, v4
    addi        t2, t2, 32
    
    vle8.v      v5, (t2)            // v5 = Weight[KH=2, KW=1]
    vmadot1     v28, v0, v5
    addi        t2, t2, 32
    
    vle8.v      v6, (t2)            // v6 = Weight[KH=2, KW=2]
    vmadot2     v28, v0, v6
    
    // Advance to next C_in block
    add         a0, a0, a4          // Act_ptr += act_C_stride
    add         a1, a1, a5          // Weight_ptr += weight_C_stride
    addi        a3, a3, -1
    bnez        a3, .L_conv_c_loop
    
    // Save Out
    vsetvli     t0, zero, e32, m2
    vse32.v     v28, (a2)

.L_conv_return:
    ret
