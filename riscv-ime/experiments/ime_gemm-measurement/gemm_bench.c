#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>

// -------------------------------------------------------------------------
// Perf Measurement Framework
// -------------------------------------------------------------------------
static int perf_fd = -1;

void init_perf() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    
    perf_fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (perf_fd == -1) {
        fprintf(stderr, "Error opening perf event. Check permissions or kernel.perf_event_paranoid.\n");
        exit(1);
    }
}

static inline void start_perf() {
    ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0);
}

static inline void stop_perf() {
    ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
}

static inline uint64_t get_cycles() {
    uint64_t count;
    if (read(perf_fd, &count, sizeof(uint64_t)) != sizeof(uint64_t)) {
        return 0;
    }
    return count;
}

void print_stats(const char *name, double *results, int n) {
    for(int i=0; i<n-1; i++) {
        for(int j=i+1; j<n; j++) {
            if(results[i] > results[j]) {
                double tmp = results[i];
                results[i] = results[j];
                results[j] = tmp;
            }
        }
    }
    double median = (n % 2 == 0) ? (results[n/2 - 1] + results[n/2]) / 2.0 : results[n/2];
    double sum = 0, sq_sum = 0;
    for(int i=0; i<n; i++) { sum += results[i]; }
    double mean = sum / n;
    for(int i=0; i<n; i++) { sq_sum += (results[i] - mean) * (results[i] - mean); }
    double stddev = sqrt(sq_sum / n);
    
    printf("  %-40s : Median %8.2f cycles/call (Stddev %6.2f)\n", name, median, stddev);
}

#define REP 10

// -------------------------------------------------------------------------
// Kernel Prototypes
// -------------------------------------------------------------------------
void ime_i8_gemm_4x8x4_i8i8i32(const int8_t *A, const int8_t *B, int32_t *C, size_t num_K, size_t A_K_stride, size_t B_K_stride);
void ime_i8_gemm_4x8x4_i8i8i32_acc(const int8_t *A, const int8_t *B, int32_t *C, size_t num_K, size_t A_K_stride, size_t B_K_stride);

void ime_i8_gemm_4x8x4_i8i8i32_rm_rm(const int8_t *A, const int8_t *B, int32_t *C, size_t num_K, size_t A_K_stride, size_t B_K_stride, size_t lda, size_t ldb);
void ime_i8_gemm_4x8x4_i8i8i32_rm_rm_acc(const int8_t *A, const int8_t *B, int32_t *C, size_t num_K, size_t A_K_stride, size_t B_K_stride, size_t lda, size_t ldb);

void ime_i8_gemm_4x8x4_i8i8i32_rm_cm(const int8_t *A, const int8_t *B, int32_t *C, size_t num_K, size_t A_K_stride, size_t B_K_stride, size_t lda, size_t ldb);
void ime_i8_gemm_4x8x4_i8i8i32_rm_cm_acc(const int8_t *A, const int8_t *B, int32_t *C, size_t num_K, size_t A_K_stride, size_t B_K_stride, size_t lda, size_t ldb);

// -------------------------------------------------------------------------
// Benchmark Macros
// -------------------------------------------------------------------------
// We pass stride 0 so it stays securely in L1 cache and doesn't exceed the allocated buffers.
#define MEASURE_GEMM_PREPACKED(NAME, FUNC, NUM_K) do { \
    long iters = 2000000 / (NUM_K > 0 ? NUM_K : 1); \
    if (iters < 1000) iters = 1000; \
    double res[REP]; \
    for (int rep=0; rep<REP; rep++) { \
        start_perf(); \
        for(long i=0; i<iters; i++) { \
            FUNC(A, B, C, NUM_K, 0, 0); \
        } \
        stop_perf(); \
        res[rep] = (double)get_cycles() / iters; \
    } \
    char buf[128]; \
    snprintf(buf, sizeof(buf), "%s (K=%d)", NAME, NUM_K); \
    print_stats(buf, res, REP); \
} while(0)

#define MEASURE_GEMM_RM(NAME, FUNC, NUM_K) do { \
    long iters = 2000000 / (NUM_K > 0 ? NUM_K : 1); \
    if (iters < 1000) iters = 1000; \
    double res[REP]; \
    for (int rep=0; rep<REP; rep++) { \
        start_perf(); \
        for(long i=0; i<iters; i++) { \
            FUNC(A, B, C, NUM_K, 0, 0, 0, 0); \
        } \
        stop_perf(); \
        res[rep] = (double)get_cycles() / iters; \
    } \
    char buf[128]; \
    snprintf(buf, sizeof(buf), "%s (K=%d)", NAME, NUM_K); \
    print_stats(buf, res, REP); \
} while(0)

int main() {
    init_perf();
    
    // Allocate 4KB buffers (always fits in L1)
    int8_t *A = NULL;
    int8_t *B = NULL;
    int32_t *C = NULL;
    
    if (posix_memalign((void**)&A, 64, 4096) != 0 ||
        posix_memalign((void**)&B, 64, 4096) != 0 ||
        posix_memalign((void**)&C, 64, 4096) != 0) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    memset(A, 1, 4096);
    memset(B, 1, 4096);
    memset(C, 0, 4096);

    int k_vals[] = {1, 10, 100};

    printf("Starting ime_gemm.S Latency Benchmark (L1 Resident)...\n");

    for (int i=0; i<3; i++) {
        int k = k_vals[i];
        printf("\n--- Sweeping num_K = %d ---\n", k);

        // 1. Pre-packed (Zero-init)
        MEASURE_GEMM_PREPACKED("1. Pre-packed (Zero-init)", ime_i8_gemm_4x8x4_i8i8i32, k);
        // 2. Pre-packed (Acc)
        MEASURE_GEMM_PREPACKED("2. Pre-packed (Accumulate)", ime_i8_gemm_4x8x4_i8i8i32_acc, k);
        
        // 3. RM-RM (Zero-init)
        MEASURE_GEMM_RM("3. RM-RM (Zero-init)", ime_i8_gemm_4x8x4_i8i8i32_rm_rm, k);
        // 4. RM-RM (Acc)
        MEASURE_GEMM_RM("4. RM-RM (Accumulate)", ime_i8_gemm_4x8x4_i8i8i32_rm_rm_acc, k);
        
        // 5. RM-CM (Zero-init)
        MEASURE_GEMM_RM("5. RM-CM (Zero-init)", ime_i8_gemm_4x8x4_i8i8i32_rm_cm, k);
        // 6. RM-CM (Acc)
        MEASURE_GEMM_RM("6. RM-CM (Accumulate)", ime_i8_gemm_4x8x4_i8i8i32_rm_cm_acc, k);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
