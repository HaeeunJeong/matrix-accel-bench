// riscv64-unknown-linux-gnu-gcc  -march=rv64gcv vmadot-gemm-demo.c -o
// gemm-vmadot-4x8x4

/*
 *
 * simple demo, using vmadot to calclate matrix multi.
 * data type of matrix A and B is int8_t.
 * data type of matrix C and CRef is int32_t.
 * A_{MxK} * B_{KxN} -> C_{MxN}
 * in this case, M fixed to 4. N fixed to 4. K fixed to 8.
 *
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#define ITER 1000000

void Reference_Gemm(size_t M, size_t N, size_t K, const int8_t *A,
                    const int8_t *B, int32_t *C) {

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < K; ++k) {
        int8_t a = A[m * K + k];
        int8_t b = B[k * N + n];
        acc += a * b;
      }
      C[m * N + n] = acc;
    }
  }
}

// B 행렬을 vmadot 연산에 적합한 메모리 레이아웃으로 재정렬(Packing)하는 함수
// 실제 산술 연산(GEMM)은 포함하지 않습니다.
void packB(size_t ROW, size_t COL, const int8_t *B, int8_t *packedB) {

  __asm__ volatile("addi         t6, zero, 8             \n\t"
                   "vsetvli      t0, zero, e8, mf8       \n\t"

                   // %= : inline assembly가 매크로처럼 여러 번 인라인될 때, 라벨 이름(LOOP_ROW)의 중복 충돌을 막기 위해 
                   // GCC가 컴파일 타임에 고유한 숫자를 자동으로 붙여주는 문법입니다.
                   "LOOP_ROW%=:                          \n\t"
                   // ROW 값을 1 감소시킵니다. 루프를 몇 번 돌지 제어하는 카운터(loop index decrement) 역할입니다.
                   "addi         %[ROW], %[ROW], -1      \n\t"

                   "LOOP_COL%=:                          \n\t"
                   "vle8.v       v0, (%[SRC])            \n\t"
                   "addi         %[SRC], %[SRC], 4       \n\t"
                   "vsse8.v      v0, (%[DST]), t6        \n\t"
                   "addi         %[DST], %[DST], 1       \n\t"

                   "bnez         %[ROW], LOOP_ROW%=      \n\t"

                   : [SRC] "+r"(B), [DST] "+r"(packedB), [ROW] "+r"(ROW)
                   : [COL] "r"(COL)
                   : "cc", "t6", "t0");
}

// A 행렬과 원본 B 행렬을 곱하는 함수 (함수 내부에서 B를 On-the-fly Packing하여 연산)
void vmadot_Gemm(size_t M, size_t N, size_t K, const int8_t *A, const int8_t *B,
                 int32_t *C) {

  int8_t packB_local[32];

  __asm__ volatile(
      // --- 1. Pack B on the fly ---
      "addi         t6, zero, 8             \n\t"
      "vsetvli      t0, zero, e8, mf8       \n\t"
      "addi         t5, zero, 8             \n\t"
      "mv           t4, %[B]                \n\t"
      "mv           t3, %[PACKB]            \n\t"

      "LOOP_ROW_FUSED%=:                    \n\t"
      "addi         t5, t5, -1              \n\t"
      "vle8.v       v0, (t4)                \n\t"
      "addi         t4, t4, 4               \n\t"
      "vsse8.v      v0, (t3), t6            \n\t"
      "addi         t3, t3, 1               \n\t"
      "bnez         t5, LOOP_ROW_FUSED%=    \n\t"

      // --- 2. vmadot ---
      "vsetvli      t0, zero, e32, m2       \n\t"
      "vxor.vv      v28, v28, v28           \n\t"

      "vsetvli      t0, zero, e8, m1        \n\t"
      "vle8.v       v0, (%[A])              \n\t"
      "vle8.v       v1, (%[PACKB])          \n\t"
      "vmadot       v28, v0, v1             \n\t"

      // --- 3. Store C ---
      "vsetvli      t0, zero, e32, m2       \n\t"
      "vse32.v      v28, (%[C])             \n\t"

      :
      : [A] "r"(A), [B] "r"(B), [C] "r"(C), [PACKB] "r"(packB_local)
      : "cc", "memory", "t0", "t3", "t4", "t5", "t6", "v0", "v1", "v28");
}

// 사전에 Packing된 행렬(packedB)을 사용하여 vmadot 행렬 연산을 수행하는 함수
void vmadot_Gemm_packB(size_t M, size_t N, size_t K, const int8_t *A, const int8_t *packedB,
                       int32_t *C) {

  __asm__ volatile(
                   // 누산기(Accumulator) 초기화: v28 레지스터를 0으로 만듦 (e32, m2 사용)
                   "vsetvli      t0, zero, e32, m2       \n\t"
                   "vxor.vv      v28, v28, v28           \n\t"

                   // 데이터 로드 및 연산을 위한 벡터 설정 (e8, m1 사용)
                   "vsetvli      t0, zero, e8, m1        \n\t"
                   // (참고: LOOP_K 라벨은 있으나 실제 bnez 분기가 없음. VLEN 범위 내에서 한 번에 처리된다고 가정)
                   "LOOP_K_PACKED%=:                     \n\t"
                   // 행렬 A 데이터를 벡터 레지스터 v0로 로드하고 포인터 32 증가
                   "vle8.v       v0, (%[A])              \n\t"
                   "addi         %[A], %[A], 32          \n\t"

                   // Packing된 행렬 B 데이터를 벡터 레지스터 v1로 로드하고 포인터 32 증가
                   "vle8.v       v1, (%[B])              \n\t"
                   "addi         %[B], %[B], 32          \n\t"

                   // v0(A)와 v1(B)의 dot-product를 수행하고 결과를 v28에 누적
                   "vmadot       v28, v0, v1             \n\t"

                   // 연산 결과를 메모리(행렬 C)에 32비트 단위로 저장
                   "vsetvli      t0, zero, e32, m2       \n\t"
                   "vse32.v      v28, (%[C])             \n\t"
                   : [A] "+r"(A), [B] "+r"(packedB), [C] "+r"(C), [M] "+r"(M)
                   : [K] "r"(K), [N] "r"(N)
                   : "cc");
}



void Test(size_t M, size_t N, const int32_t *Ref, const int32_t *Real) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      assert(Ref[m * N + n] == Real[m * N + n]);
    }
  }
}

void display(size_t row, size_t col, void *output, size_t t) {
  if (t == 1) {
    int8_t *C = (int8_t *)output;
    for (size_t i = 0; i < row; ++i) {
      for (size_t j = 0; j < col; ++j) {
        printf("%5d ", C[i * col + j]);
      }
      printf("\n");
    }
  } else if (t == 4) {
    int32_t *C = (int32_t *)output;
    for (size_t i = 0; i < row; ++i) {
      for (size_t j = 0; j < col; ++j) {
        printf("%7d ", C[i * col + j]);
      }
      printf("\n");
    }
  }
}

int main() {
  setbuf(stdout, NULL);
  int8_t A[32] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                  0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};

  int8_t B[32] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

  srand((uint32_t)time(NULL));
  for (size_t index = 0; index < 32; ++index) {
    A[index] = rand() % 256 - 128;
    B[index] = rand() % 256 - 128;
  }

  int8_t packedB[32] = {0};
  int32_t C[16] = {0};
  int32_t CRef[16] = {0};

  uint64_t t0, t1;

  printf("\n");
  printf(
      "------------------------------------------------------------------\n");
  printf(" %-30s | %-14s | %s\n", "Function Name", "Execution Time",
         "Verification");
  printf(
      "------------------------------------------------------------------\n");

  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    packB(8, 4, B, packedB);
  }
  t1 = now_ns();
  printf(" %-30s | %10.3f ns   | %s\n", "packB (Overhead)",
         (double)(t1 - t0) / ITER, "N/A");

  // 1. Reference GEMM
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Reference_Gemm(4, 4, 8, A, B, CRef);
  }
  t1 = now_ns();
  printf(" %-30s | %10.3f ns   | %s\n", "Reference_Gemm",
         (double)(t1 - t0) / ITER, "Baseline");

  // 2. vmadot_Gemm (Raw A and B, on-the-fly packing)
  for (int i = 0; i < 16; i++)
    C[i] = 0;
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    vmadot_Gemm(4, 4, 8, A, B, C);
  }
  t1 = now_ns();
  Test(4, 4, CRef, C);
  printf(" %-30s | %10.3f ns   | %s\n", "vmadot_Gemm (Fused)", (double)(t1 - t0) / ITER,
         "Passed");

  // 3. vmadot_Gemm_packB (A and packedB)
  for (int i = 0; i < 16; i++)
    C[i] = 0;
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    vmadot_Gemm_packB(4, 4, 8, A, packedB, C);
  }
  t1 = now_ns();
  Test(4, 4, CRef, C);
  printf(" %-30s | %10.3f ns   | %s\n", "vmadot_Gemm_packB", (double)(t1 - t0) / ITER,
         "Passed");

  printf(
      "------------------------------------------------------------------\n");

  printf("*********************************\n");
  printf("matrix A: \n");
  display(4, 8, A, 1);
  printf("matrix B: \n");
  display(8, 4, B, 1);
  printf("matrix packB: \n");
  display(4, 8, packedB, 1);

  printf("matrix CRef: \n");
  display(4, 4, CRef, 4);

  printf("matrix C: \n");
  display(4, 4, C, 4);
  printf("*********************************\n");
  return 0;
}
