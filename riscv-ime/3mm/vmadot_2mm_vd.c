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

#define ITER 50000

#define Ti 4
#define Tk 8
#define Tj 4

#define LOGICAL(i, j, STRIDE) ((i) * (STRIDE) + (j))

#define A_VIRTUAL(i, k, K)                                                     \
  (((i) / Ti) * ((K) / Tk) * (Ti * Tk) + ((k) / Tk) * (Ti * Tk) +              \
   ((i) % Ti) * Tk + ((k) % Tk))

#define B_VIRTUAL_TRANSPOSED(k, n, N)                                          \
  (((k) / Tk) * ((N) / Tj) * (Tk * Tj) + ((n) / Tj) * (Tk * Tj) +              \
   ((n) % Tj) * Tk + ((k) % Tk))

#define C_VIRTUAL(i, n, N)                                                     \
  (((i) / Ti) * ((N) / Tj) * (Ti * Tj) + ((n) / Tj) * (Ti * Tj) +              \
   ((i) % Ti) * Tj + ((n) % Tj))

enum StoreLayout {
  STORE_ROW_MAJOR_INT8,
  STORE_ROW_MAJOR_INT32,
  STORE_VD_A_INT8,
  STORE_VD_C_INT32
};

// =========================================================================
// 1. Scalar Naive 2MM
// =========================================================================
void Scalar_Naive_Gemm(size_t M, size_t N, size_t O, const int8_t *A,
                       const int8_t *B, const int8_t *D, int8_t *C,
                       int32_t *E) {
  // C = A * B
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      int32_t acc = 0;
      for (size_t k = 0; k < N; k++) {
        acc += A[LOGICAL(m, k, N)] * B[LOGICAL(k, n, N)];
      }
      C[LOGICAL(m, n, N)] = (int8_t)acc;
    }
  }
  // E = C * D
  for (size_t m = 0; m < M; m++) {
    for (size_t o = 0; o < O; o++) {
      int32_t acc = 0;
      for (size_t n = 0; n < N; n++) {
        acc += C[LOGICAL(m, n, N)] * D[LOGICAL(n, o, O)];
      }
      E[LOGICAL(m, o, O)] = acc;
    }
  }
}

// =========================================================================
// Core GEMM Implementations for Cases 2, 3, 4
// =========================================================================

void Gemm_Baseline_vmadot(size_t M, size_t N, size_t K, const int8_t *A_rm,
                          const int8_t *B_rm, void *C_out,
                          enum StoreLayout layout) {
  size_t num_I = M / Ti;
  size_t num_J = N / Tj;
  size_t num_K = K / Tk;

  for (size_t I = 0; I < num_I; ++I) {
    for (size_t J = 0; J < num_J; ++J) {
      int32_t C_tile[Ti * Tj] = {0};

      for (size_t K_tile = 0; K_tile < num_K; ++K_tile) {
        int8_t packA[Ti * Tk];
        for (int i = 0; i < Ti; i++)
          for (int k = 0; k < Tk; k++)
            packA[i * Tk + k] = A_rm[LOGICAL(I * Ti + i, K_tile * Tk + k, K)];

        int8_t packB[Tk * Tj];
        for (int k = 0; k < Tk; k++)
          for (int n = 0; n < Tj; n++)
            packB[n * Tk + k] = B_rm[LOGICAL(K_tile * Tk + k, J * Tj + n, N)];

        __asm__ volatile(
            // 누산기 C 타일 불러오기 및 초기화
            "vsetvli      t0, zero, e32, m2 \n\t"
            "vle32.v      v28, (%[C])       \n\t"
            // Packing된 A, B 행렬 타일 로드 및 연산
            "vsetvli      t0, zero, e8, m1  \n\t"
            "vle8.v       v0, (%[A])        \n\t"
            "vle8.v       v1, (%[B])        \n\t"
            "vmadot       v28, v0, v1       \n\t"
            // 연산 결과를 C 타일에 다시 저장
            "vsetvli      t0, zero, e32, m2 \n\t"
            "vse32.v      v28, (%[C])       \n\t"
            :
            : [A] "r"(packA), [B] "r"(packB), [C] "r"(C_tile)
            : "cc", "memory", "t0", "v0", "v1", "v28");
      }

      if (layout == STORE_ROW_MAJOR_INT8) {
        int8_t *out = (int8_t *)C_out;
        for (int i = 0; i < Ti; i++)
          for (int j = 0; j < Tj; j++)
            out[LOGICAL(I * Ti + i, J * Tj + j, N)] =
                (int8_t)C_tile[i * Tj + j];
      } else if (layout == STORE_ROW_MAJOR_INT32) {
        int32_t *out = (int32_t *)C_out;
        for (int i = 0; i < Ti; i++)
          for (int j = 0; j < Tj; j++)
            out[LOGICAL(I * Ti + i, J * Tj + j, N)] = C_tile[i * Tj + j];
      }
    }
  }
}

void Gemm_HalfVD_vmadot(size_t M, size_t N, size_t K, const int8_t *A_rm,
                        const int8_t *B_vd, void *C_out,
                        enum StoreLayout layout) {
  size_t num_I = M / Ti;
  size_t num_J = N / Tj;
  size_t num_K = K / Tk;

  for (size_t I = 0; I < num_I; ++I) {
    for (size_t J = 0; J < num_J; ++J) {
      int32_t C_tile[Ti * Tj] = {0};

      for (size_t K_tile = 0; K_tile < num_K; ++K_tile) {
        int8_t packA[Ti * Tk];
        for (int i = 0; i < Ti; i++)
          for (int k = 0; k < Tk; k++)
            packA[i * Tk + k] = A_rm[LOGICAL(I * Ti + i, K_tile * Tk + k, K)];

        const int8_t *B_tile_start = &B_vd[(K_tile * num_J + J) * (Tk * Tj)];

        __asm__ volatile(
            // 누산기 C 타일 불러오기 및 초기화
            "vsetvli      t0, zero, e32, m2 \n\t"
            "vle32.v      v28, (%[C])       \n\t"
            // Packing된 A 타일과 VD로 레이아웃된 B 타일 로드 및 연산
            "vsetvli      t0, zero, e8, m1  \n\t"
            "vle8.v       v0, (%[A])        \n\t"
            "vle8.v       v1, (%[B])        \n\t"
            "vmadot       v28, v0, v1       \n\t"
            // 연산 결과를 C 타일에 다시 저장
            "vsetvli      t0, zero, e32, m2 \n\t"
            "vse32.v      v28, (%[C])       \n\t"
            :
            : [A] "r"(packA), [B] "r"(B_tile_start), [C] "r"(C_tile)
            : "cc", "memory", "t0", "v0", "v1", "v28");
      }

      if (layout == STORE_ROW_MAJOR_INT32) {
        int32_t *out = (int32_t *)C_out;
        for (int i = 0; i < Ti; i++)
          for (int j = 0; j < Tj; j++)
            out[LOGICAL(I * Ti + i, J * Tj + j, N)] = C_tile[i * Tj + j];
      }
    }
  }
}

void Gemm_VD_vmadot_Fast(size_t M, size_t N, size_t K, const int8_t *A_vd,
                         const int8_t *B_vd, void *C_out,
                         enum StoreLayout layout) {
  size_t num_I = M / Ti;
  size_t num_J = N / Tj;
  size_t num_K = K / Tk;

  for (size_t I = 0; I < num_I; ++I) {
    for (size_t J = 0; J < num_J; ++J) {
      int32_t C_tile[Ti * Tj] = {0};
      const int8_t *A_tile_start = &A_vd[I * num_K * (Ti * Tk)];
      const int8_t *B_tile_start = &B_vd[J * (Tk * Tj)];
      size_t B_stride = num_J * (Tk * Tj);

      __asm__ volatile(
          // 누산기(v28) 0으로 초기화
          "vsetvli      t0, zero, e32, m2 \n\t"
          "vxor.vv      v28, v28, v28     \n\t"
          // 행렬 A, B 타일 로드 및 연산 환경 설정
          "vsetvli      t0, zero, e8, m1  \n\t"
          "mv           t1, %[num_K]      \n\t" // K 타일 루프 카운터
          "mv           t2, %[A]          \n\t" // A 타일 주소
          "mv           t3, %[B]          \n\t" // B 타일 주소
          "1:                             \n\t" // 로컬 라벨 (루프 시작)
          "vle8.v       v0, (t2)          \n\t" // A 타일 로드 (Ti*Tk)
          "vle8.v       v1, (t3)          \n\t" // B 타일 로드 (Tk*Tj)
          "vmadot       v28, v0, v1       \n\t" // dot-product 연산
          "add          t2, t2, %[stride_A] \n\t" // 다음 A 타일로 주소 이동 (+stride_A)
          "add          t3, t3, %[B_str]  \n\t" // 다음 B 타일로 주소 이동
                                                // (+stride)
          "addi         t1, t1, -1        \n\t" // 루프 카운터 1 감소
          "bnez         t1, 1b            \n\t" // 카운터가 0이 아니면 1번
                                                // 라벨로 반복 분기 (backward)
          // 연산된 C 타일 결과 저장
          "vsetvli      t0, zero, e32, m2 \n\t"
          "vse32.v      v28, (%[C])       \n\t"
          :
          : [A] "r"(A_tile_start), [B] "r"(B_tile_start), [C] "r"(C_tile),
            [num_K] "r"(num_K), [B_str] "r"(B_stride), [stride_A] "r"(Ti * Tk)
          : "cc", "memory", "t0", "t1", "t2", "t3", "v0", "v1", "v28");

      if (layout == STORE_ROW_MAJOR_INT8) {
        int8_t *out = (int8_t *)C_out;
        for (int i = 0; i < Ti; i++)
          for (int j = 0; j < Tj; j++)
            out[LOGICAL(I * Ti + i, J * Tj + j, N)] =
                (int8_t)C_tile[i * Tj + j];
      } else if (layout == STORE_VD_A_INT8) {
        int8_t *out = (int8_t *)C_out;
        size_t K_tile = J / 2;
        size_t k_rem = (J % 2) * Tj;
        size_t base_idx = I * (N / Tk) * (Ti * Tk) + K_tile * (Ti * Tk) + k_rem;
        for (int i = 0; i < Ti; i++) {
          for (int j = 0; j < Tj; j++) {
            out[base_idx + i * Tk + j] = (int8_t)C_tile[i * Tj + j];
          }
        }
      } else if (layout == STORE_VD_C_INT32) {
        int32_t *out = (int32_t *)C_out;
        size_t base_idx = (I * num_J + J) * (Ti * Tj);
        for (int i = 0; i < Ti; i++) {
          for (int j = 0; j < Tj; j++) {
            out[base_idx + i * Tj + j] = C_tile[i * Tj + j];
          }
        }
      }
    }
  }
}

// =========================================================================
// Runners
// =========================================================================

// 2. Baseline HW
void Run_Case2_Baseline(size_t M, size_t N, size_t O, const int8_t *A,
                        const int8_t *B, const int8_t *D, int8_t *C,
                        int32_t *E) {
  Gemm_Baseline_vmadot(M, N, N, A, B, C, STORE_ROW_MAJOR_INT8);
  Gemm_Baseline_vmadot(M, O, N, C, D, E, STORE_ROW_MAJOR_INT32);
}

// 3. VD Local (VD_Tiled_Transposed)
void Run_Case3_VD_Local(size_t M, size_t N, size_t O, const int8_t *A_vd,
                        const int8_t *B_vd, const int8_t *D_vd, int8_t *C_rm,
                        int32_t *E_rm) {
  Gemm_VD_vmadot_Fast(M, N, N, A_vd, B_vd, C_rm, STORE_ROW_MAJOR_INT8);
  Gemm_HalfVD_vmadot(M, O, N, C_rm, D_vd, E_rm, STORE_ROW_MAJOR_INT32);
}

// 4. VD Global (VD_Tiled_Transposed_Global)
void Run_Case4_VD_Global(size_t M, size_t N, size_t O, const int8_t *A_vd,
                         const int8_t *B_vd, const int8_t *D_vd, int8_t *C_vd,
                         int32_t *E_vd) {
  Gemm_VD_vmadot_Fast(M, N, N, A_vd, B_vd, C_vd, STORE_VD_A_INT8);
  Gemm_VD_vmadot_Fast(M, O, N, C_vd, D_vd, E_vd, STORE_VD_C_INT32);
}

// =========================================================================
// Testing Utilities
// =========================================================================

void Test_Logical(size_t M, size_t N, const int32_t *Ref, const int32_t *Real) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t n = 0; n < N; ++n) {
      if (Ref[LOGICAL(i, n, N)] != Real[LOGICAL(i, n, N)]) {
        printf("Mismatch Logical at (%zu, %zu)!\n", i, n);
        assert(0);
      }
    }
  }
}

void Test_VD(size_t M, size_t N, const int32_t *Ref, const int32_t *Real_vd) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t n = 0; n < N; ++n) {
      if (Ref[LOGICAL(i, n, N)] != Real_vd[C_VIRTUAL(i, n, N)]) {
        printf("Mismatch VD at (%zu, %zu)!\n", i, n);
        assert(0);
      }
    }
  }
}

int main() {
  setbuf(stdout, NULL);

  size_t M = 16, N = 16, O = 16;
  size_t K = 16; // C=AxB where K=16

  int8_t *A = (int8_t *)malloc(M * N);
  int8_t *B = (int8_t *)malloc(N * N);
  int8_t *D = (int8_t *)malloc(N * O);

  int8_t *C_ref = (int8_t *)malloc(M * N);
  int32_t *E_ref = (int32_t *)malloc(M * O * sizeof(int32_t));

  int8_t *A_vd = (int8_t *)malloc(M * N);
  int8_t *B_vd = (int8_t *)malloc(N * N);
  int8_t *D_vd = (int8_t *)malloc(N * O);

  srand((uint32_t)time(NULL));

  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < N; ++k) {
      int8_t val = rand() % 32 - 16;
      A[LOGICAL(i, k, N)] = val;
      A_vd[A_VIRTUAL(i, k, N)] = val;
    }
  }

  for (size_t k = 0; k < N; ++k) {
    for (size_t n = 0; n < N; ++n) {
      int8_t val = rand() % 32 - 16;
      B[LOGICAL(k, n, N)] = val;
      B_vd[B_VIRTUAL_TRANSPOSED(k, n, N)] = val;

      int8_t d_val = rand() % 32 - 16;
      D[LOGICAL(k, n, O)] = d_val;
      D_vd[B_VIRTUAL_TRANSPOSED(k, n, O)] = d_val;
    }
  }

  uint64_t t0, t1;
  printf("\n");
  printf(
      "==================================================================\n");
  printf(" 3MM Benchmark (C=AxB, E=CxD) | Size: M=%zu, N=%zu, O=%zu\n", M, N,
         O);
  printf(
      "==================================================================\n");
  printf(" %-30s | %-14s | %s\n", "Function Name", "Execution Time",
         "Verification");
  printf(
      "------------------------------------------------------------------\n");

  // 1. Scalar Naive 2MM
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Scalar_Naive_Gemm(M, N, O, A, B, D, C_ref, E_ref);
  }
  t1 = now_ns();
  printf(" %-30s | %10.3f ns   | %s\n", "1. Scalar_Naive_2MM",
         (double)(t1 - t0) / ITER, "Baseline");

  // 2. Baseline HW vmadot
  int8_t *C_rm = (int8_t *)malloc(M * N);
  int32_t *E_rm = (int32_t *)malloc(M * O * sizeof(int32_t));

  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Run_Case2_Baseline(M, N, O, A, B, D, C_rm, E_rm);
  }
  t1 = now_ns();
  Test_Logical(M, O, E_ref, E_rm);
  printf(" %-30s | %10.3f ns   | %s\n", "2. Baseline_HW_vmadot",
         (double)(t1 - t0) / ITER, "Passed");

  // 3. VD Tiled Transposed (Local)
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Run_Case3_VD_Local(M, N, O, A_vd, B_vd, D_vd, C_rm, E_rm);
  }
  t1 = now_ns();
  Test_Logical(M, O, E_ref, E_rm);
  printf(" %-30s | %10.3f ns   | %s\n", "3. VD_Tiled_Transposed",
         (double)(t1 - t0) / ITER, "Passed");

  // 4. VD Tiled Transposed Global
  int8_t *C_vd = (int8_t *)malloc(M * N);
  int32_t *E_vd = (int32_t *)malloc(M * O * sizeof(int32_t));

  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Run_Case4_VD_Global(M, N, O, A_vd, B_vd, D_vd, C_vd, E_vd);
  }
  t1 = now_ns();
  Test_VD(M, O, E_ref, E_vd);
  printf(" %-30s | %10.3f ns   | %s\n", "4. VD_Tiled_Transposed_Global",
         (double)(t1 - t0) / ITER, "Passed");

  printf(
      "==================================================================\n");

  free(A);
  free(B);
  free(D);
  free(C_ref);
  free(E_ref);
  free(A_vd);
  free(B_vd);
  free(D_vd);
  free(C_rm);
  free(E_rm);
  free(C_vd);
  free(E_vd);

  return 0;
}
