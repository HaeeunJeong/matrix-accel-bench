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

#define ITER 100000

#define Ti 4
#define Tk 8
#define Tj 4

// 1. 일반적인 Row-major 논리적 매핑
#define A_LOGICAL(i, k, K) ((i) * (K) + (k))
#define B_LOGICAL(k, n, N) ((k) * (N) + (n))
#define C_LOGICAL(i, n, N) ((i) * (N) + (n))

// 2. Tile-aware Virtual Dimension 매핑
#define A_VIRTUAL(i, k, K) \
  (((i) / Ti) * ((K) / Tk) * (Ti * Tk) + \
   ((k) / Tk) * (Ti * Tk) + \
   ((i) % Ti) * Tk + \
   ((k) % Tk))

// 3. Tile-aware + Transposed Virtual Dimension 매핑 (가장 핵심!)
// B 행렬은 Tile 간에는 정상 순서지만, Tile 내부에서는 열(Column) 방향으로 연속되도록 저장합니다.
#define B_VIRTUAL_TRANSPOSED(k, n, N) \
  (((k) / Tk) * ((N) / Tj) * (Tk * Tj) + \
   ((n) / Tj) * (Tk * Tj) + \
   ((n) % Tj) * Tk + \
   ((k) % Tk))

#define C_VIRTUAL(i, n, N) \
  (((i) / Ti) * ((N) / Tj) * (Ti * Tj) + \
   ((n) / Tj) * (Ti * Tj) + \
   ((i) % Ti) * Tj + \
   ((n) % Tj))

void Scalar_Naive_Gemm(size_t M, size_t N, size_t K, const int8_t *A,
                       const int8_t *B, int32_t *C) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < K; ++k) {
        int8_t a = A[A_LOGICAL(m, k, K)];
        int8_t b = B[B_LOGICAL(k, n, N)];
        acc += a * b;
      }
      C[C_LOGICAL(m, n, N)] = acc;
    }
  }
}

// Baseline: Row-Major 메모리 구조에서 vmadot을 쓰기 위해 런타임에 억지로 타일을 복사/Transpose (Packing)
void Gemm_Baseline_vmadot(size_t M, size_t N, size_t K, const int8_t *A,
                          const int8_t *B, int32_t *C) {
  size_t num_I = M / Ti;
  size_t num_J = N / Tj;
  size_t num_K = K / Tk;

  for (size_t I = 0; I < num_I; ++I) {
    for (size_t J = 0; J < num_J; ++J) {
      int32_t C_tile[16] = {0};

      for (size_t K_tile = 0; K_tile < num_K; ++K_tile) {
        // 1. Pack A tile (4x8) - Row major 2D -> 1D contiguous
        int8_t packA[32];
        for (int i = 0; i < Ti; i++) {
          for (int k = 0; k < Tk; k++) {
            packA[i * Tk + k] = A[A_LOGICAL(I * Ti + i, K_tile * Tk + k, K)];
          }
        }
        
        // 2. Pack B tile (8x4) -> Transpose to (4x8) contiguous
        int8_t packB[32];
        for (int k = 0; k < Tk; k++) {
          for (int n = 0; n < Tj; n++) {
            packB[n * Tk + k] = B[B_LOGICAL(K_tile * Tk + k, J * Tj + n, N)];
          }
        }

        // 3. vmadot execution
        __asm__ volatile(
            // 누산기(v28) 초기화
            "vsetvli      t0, zero, e32, m2 \n\t"
            "vle32.v      v28, (%[C])       \n\t"
            // 행렬 A, B 로드 및 연산
            "vsetvli      t0, zero, e8, m1  \n\t"
            "vle8.v       v0, (%[A])        \n\t"
            "vle8.v       v1, (%[B])        \n\t"
            "vmadot       v28, v0, v1       \n\t"
            // 결과 C 타일 저장
            "vsetvli      t0, zero, e32, m2 \n\t"
            "vse32.v      v28, (%[C])       \n\t"
            :
            : [A] "r"(packA), [B] "r"(packB), [C] "r"(C_tile)
            : "cc", "memory", "t0", "v0", "v1", "v28");
      }

      // Store packed C_tile back to Row-Major C array
      for (int i = 0; i < Ti; i++) {
        for (int n = 0; n < Tj; n++) {
          C[C_LOGICAL(I * Ti + i, J * Tj + n, N)] = C_tile[i * Tj + n];
        }
      }
    }
  }
}

// 완전히 Tiling + Transpose가 적용된 Virtual Dimension을 활용하는 
// Zero-Overhead vmadot GEMM 구현
void Gemm_VD_Tiled_Transposed_vmadot(size_t M, size_t N, size_t K, const int8_t *A_vd,
                                     const int8_t *B_vd, int32_t *C_vd) {
  size_t num_I = M / Ti;
  size_t num_J = N / Tj;
  size_t num_K = K / Tk;

  for (size_t I = 0; I < num_I; ++I) {
    for (size_t J = 0; J < num_J; ++J) {
      // 해당 C 타일의 포인터
      int32_t *C_tile = &C_vd[(I * num_J + J) * (Ti * Tj)];

      // C 타일 초기화
      for (int x = 0; x < Ti * Tj; x++) {
        C_tile[x] = 0;
      }

      const int8_t *A_tile_start = &A_vd[I * num_K * (Ti * Tk)];
      const int8_t *B_tile_start = &B_vd[J * (Tk * Tj)]; // j tile is contiguous within B_vd layout
      // Wait, B_vd address calculation: 
      // K_tile varies. So B_tile address = (K_tile * num_J + J) * 32.
      // So B_stride = num_J * 32.

      size_t B_stride = num_J * (Tk * Tj);

      // Inline Assembly for computing 1 tile of C (4x4)
      __asm__ volatile(
          // 누산기(v28) 0으로 초기화
          "vsetvli      t0, zero, e32, m2 \n\t"
          "vxor.vv      v28, v28, v28     \n\t"   // v28 = Accumulator 초기화

          "vsetvli      t0, zero, e8, m1  \n\t"
          "mv           t1, %[num_K]      \n\t"   // t1에 K 타일 루프 카운터(num_K) 설정
          "mv           t2, %[A]          \n\t"   // A 타일의 시작 주소
          "mv           t3, %[B]          \n\t"   // B 타일의 시작 주소

          "LOOP_K_TILE%=:                 \n\t"
          "vle8.v       v0, (t2)          \n\t"   // A 타일 로드 (한 번에 32바이트)
          "vle8.v       v1, (t3)          \n\t"   // B 타일 로드 (VD 덕분에 패킹 없이 32바이트 로드!)
          "vmadot       v28, v0, v1       \n\t"   // dot-product 연산
          
          "addi         t2, t2, 32        \n\t"   // 다음 A 타일 주소 계산 (32바이트 이동)
          "add          t3, t3, %[B_str]  \n\t"   // 다음 B 타일 주소 계산 (B_stride 만큼 이동)
          "addi         t1, t1, -1        \n\t"   // 루프 카운터 1 감소
          "bnez         t1, LOOP_K_TILE%= \n\t"   // 카운터가 0이 아니면 루프 반복

          // 최종 연산된 C 타일 결과 저장
          "vsetvli      t0, zero, e32, m2 \n\t"
          "vse32.v      v28, (%[C])       \n\t"   // 최종 C 타일 저장
          :
          : [A] "r"(A_tile_start), [B] "r"(B_tile_start), [C] "r"(C_tile),
            [num_K] "r"(num_K), [B_str] "r"(B_stride)
          : "cc", "memory", "t0", "t1", "t2", "t3", "v0", "v1", "v28");
    }
  }
}

void Test_Logical(size_t M, size_t N, const int32_t *Ref, const int32_t *Real) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t n = 0; n < N; ++n) {
      int32_t ref_val = Ref[C_LOGICAL(i, n, N)];
      int32_t real_val = Real[C_LOGICAL(i, n, N)];
      if (ref_val != real_val) {
          printf("Mismatch at (%zu, %zu)! Ref: %d, Real: %d\n", i, n, ref_val, real_val);
          assert(0);
      }
    }
  }
}

// 검증을 위해 논리적인 주소 공간에서 비교하는 함수 (VD 레이아웃용)
void Test_VD(size_t M, size_t N, const int32_t *Ref, const int32_t *Real_vd) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t n = 0; n < N; ++n) {
      int32_t ref_val = Ref[C_LOGICAL(i, n, N)];
      int32_t real_val = Real_vd[C_VIRTUAL(i, n, N)];
      if (ref_val != real_val) {
          printf("Mismatch at (%zu, %zu)! Ref: %d, Real: %d\n", i, n, ref_val, real_val);
          assert(0);
      }
    }
  }
}

int main() {
  setbuf(stdout, NULL);

  size_t M = 16;
  size_t N = 16;
  size_t K = 16;

  int8_t *A = (int8_t *)malloc(M * K);
  int8_t *B = (int8_t *)malloc(K * N);
  int32_t *CRef = (int32_t *)malloc(M * N * sizeof(int32_t));

  int8_t *A_vd = (int8_t *)malloc(M * K);
  int8_t *B_vd = (int8_t *)malloc(K * N);
  int32_t *C_vd = (int32_t *)malloc(M * N * sizeof(int32_t));

  srand((uint32_t)time(NULL));
  
  // 데이터 초기화: Logical 배열과 Virtual Dimension 배열을 동시에 채움
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      int8_t val = rand() % 256 - 128;
      A[A_LOGICAL(i, k, K)] = val;
      A_vd[A_VIRTUAL(i, k, K)] = val;
    }
  }

  for (size_t k = 0; k < K; ++k) {
    for (size_t n = 0; n < N; ++n) {
      int8_t val = rand() % 256 - 128;
      B[B_LOGICAL(k, n, N)] = val;
      B_vd[B_VIRTUAL_TRANSPOSED(k, n, N)] = val;
    }
  }

  uint64_t t0, t1;
  printf("\n");
  printf("------------------------------------------------------------------\n");
  printf(" Matrix Size: M=%zu, N=%zu, K=%zu\n", M, N, K);
  printf("------------------------------------------------------------------\n");
  printf(" %-30s | %-14s | %s\n", "Function Name", "Execution Time", "Verification");
  printf("------------------------------------------------------------------\n");

  // 1. Scalar Naive GEMM
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Scalar_Naive_Gemm(M, N, K, A, B, CRef);
  }
  t1 = now_ns();
  printf(" %-30s | %10.3f ns   | %s\n", "Scalar_Naive_Gemm",
         (double)(t1 - t0) / ITER, "Baseline");

  // 2. Baseline HW GEMM (On-the-fly packing)
  int32_t *C_baseline = (int32_t *)malloc(M * N * sizeof(int32_t));
  for (size_t x = 0; x < M * N; x++) C_baseline[x] = 0;
  
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Gemm_Baseline_vmadot(M, N, K, A, B, C_baseline);
  }
  t1 = now_ns();
  Test_Logical(M, N, CRef, C_baseline);
  printf(" %-30s | %10.3f ns   | %s\n", "Baseline_HW_vmadot (Packed)",
         (double)(t1 - t0) / ITER, "Passed");

  // 3. Gemm VD Tiled Transposed vmadot
  t0 = now_ns();
  for (int i = 0; i < ITER; ++i) {
    Gemm_VD_Tiled_Transposed_vmadot(M, N, K, A_vd, B_vd, C_vd);
  }
  t1 = now_ns();
  Test_VD(M, N, CRef, C_vd);
  printf(" %-30s | %10.3f ns   | %s\n", "Gemm_VD_Tiled_Transposed",
         (double)(t1 - t0) / ITER, "Passed");

  printf("------------------------------------------------------------------\n");

  free(A); free(B); free(CRef); free(C_baseline);
  free(A_vd); free(B_vd); free(C_vd);

  return 0;
}
