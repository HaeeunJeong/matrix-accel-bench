#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>

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
    
    printf("  %-28s : Median %6.2f cycles/inst (Stddev %5.2f)\n", name, median, stddev);
}

#define REP 10

// TP (Throughput) ASM BLOCKS
// Destination rotation to prevent WAW: v0 -> v1 -> v2 -> v3
#define VLE8_TP_BLK \
    "vle8.v v0, (%0)\n\tadd %0, %0, %2\n\t" \
    "vle8.v v1, (%0)\n\tadd %0, %0, %2\n\t" \
    "vle8.v v2, (%0)\n\tadd %0, %0, %2\n\t" \
    "vle8.v v3, (%0)\n\tadd %0, %0, %2"
#define VLE8_TP_SGL "vle8.v v0, (%0)\n\tadd %0, %0, %2"

#define VLSE8_TP_BLK \
    "vlse8.v v0, (%0), %1\n\tadd %0, %0, %2\n\t" \
    "vlse8.v v1, (%0), %1\n\tadd %0, %0, %2\n\t" \
    "vlse8.v v2, (%0), %1\n\tadd %0, %0, %2\n\t" \
    "vlse8.v v3, (%0), %1\n\tadd %0, %0, %2"
#define VLSE8_TP_SGL "vlse8.v v0, (%0), %1\n\tadd %0, %0, %2"

#define VLUXEI32_TP_BLK \
    "vluxei32.v v0, (%0), v4\n\tadd %0, %0, %2\n\t" \
    "vluxei32.v v1, (%0), v4\n\tadd %0, %0, %2\n\t" \
    "vluxei32.v v2, (%0), v4\n\tadd %0, %0, %2\n\t" \
    "vluxei32.v v3, (%0), v4\n\tadd %0, %0, %2"
#define VLUXEI32_TP_SGL "vluxei32.v v0, (%0), v4\n\tadd %0, %0, %2"

#define VLOXEI32_TP_BLK \
    "vloxei32.v v0, (%0), v4\n\tadd %0, %0, %2\n\t" \
    "vloxei32.v v1, (%0), v4\n\tadd %0, %0, %2\n\t" \
    "vloxei32.v v2, (%0), v4\n\tadd %0, %0, %2\n\t" \
    "vloxei32.v v3, (%0), v4\n\tadd %0, %0, %2"
#define VLOXEI32_TP_SGL "vloxei32.v v0, (%0), v4\n\tadd %0, %0, %2"

// LATENCY ASM BLOCKS
// vmv.x.s only baseline to separate scalar overhead
#define VMV_ONLY_LAT_BLK \
    "vmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"
#define VMV_ONLY_LAT_SGL "vmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"

#define VLE8_LAT_BLK \
    "vle8.v v0, (%0)\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vle8.v v0, (%0)\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vle8.v v0, (%0)\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vle8.v v0, (%0)\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"
#define VLE8_LAT_SGL "vle8.v v0, (%0)\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"

#define VLSE8_LAT_BLK \
    "vlse8.v v0, (%0), %1\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vlse8.v v0, (%0), %1\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vlse8.v v0, (%0), %1\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vlse8.v v0, (%0), %1\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"
#define VLSE8_LAT_SGL "vlse8.v v0, (%0), %1\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"

#define VLUXEI32_LAT_BLK \
    "vluxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vluxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vluxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vluxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"
#define VLUXEI32_LAT_SGL "vluxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"

#define VLOXEI32_LAT_BLK \
    "vloxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vloxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vloxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1\n\t" \
    "vloxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"
#define VLOXEI32_LAT_SGL "vloxei32.v v0, (%0), v4\n\tvmv.x.s t1, v0\n\tadd t1, t1, %2\n\tadd %0, %0, t1"

#define SETUP_CONTIG \
    asm volatile("li a0, 32\n\tvsetvli t0, a0, e32, m4\n\tvle32.v v4, (%0)\n\tvsetvli t0, a0, e8, m1" :: "r"(idx_contig) : "t0", "v4", "v5", "v6", "v7", "a0");

#define SETUP_STRIDED \
    asm volatile("li a0, 32\n\tvsetvli t0, a0, e32, m4\n\tvle32.v v4, (%0)\n\tvsetvli t0, a0, e8, m1" :: "r"(idx_strided) : "t0", "v4", "v5", "v6", "v7", "a0");

#define SETUP_NONE \
    asm volatile("li a0, 32\n\tvsetvli t0, a0, e8, m1" ::: "t0", "a0");

// Measure Macro
#define MEASURE(NAME, ASM_BLK, ASM_SGL, IDX_SETUP, MAX_OFFSET, ADVANCE) do { \
    long advance = ADVANCE; \
    long max_offset = MAX_OFFSET; \
    long loads_per_pass = (W - max_offset) / advance; \
    if (loads_per_pass <= 0) loads_per_pass = 1; \
    long iters = 2000000 / loads_per_pass; \
    if (iters <= 0) iters = 1; \
    long total_inst = iters * loads_per_pass; \
    double res[REP]; \
    for (int rep=0; rep<REP; rep++) { \
        start_perf(); \
        IDX_SETUP \
        for (long i = 0; i < iters; i++) { \
            int8_t *ptr = A; \
            long count = loads_per_pass; \
            while (count >= 16) { \
                asm volatile( \
                    ASM_BLK "\n\t" ASM_BLK "\n\t" \
                    ASM_BLK "\n\t" ASM_BLK \
                    : "+r"(ptr) : "r"(stride), "r"(advance) : "v0", "v1", "v2", "v3", "t1" \
                ); \
                count -= 16; \
            } \
            while (count >= 4) { \
                asm volatile(ASM_BLK : "+r"(ptr) : "r"(stride), "r"(advance) : "v0", "v1", "v2", "v3", "t1"); \
                count -= 4; \
            } \
            while (count > 0) { \
                asm volatile(ASM_SGL : "+r"(ptr) : "r"(stride), "r"(advance) : "v0", "t1"); \
                count--; \
            } \
        } \
        stop_perf(); \
        res[rep] = (double)get_cycles() / total_inst; \
    } \
    print_stats(NAME, res, REP); \
} while(0)


void run_sweep(int W, int8_t *A, uint32_t *idx_contig, uint32_t *idx_strided) {
    long stride = 64;
    printf("\n=== Working Set: %d KB ===\n", W / 1024);
    
    // VMV Baseline: Subtract this from LAT to get pure load latency
    MEASURE("vmv.x.s only LAT", VMV_ONLY_LAT_BLK, VMV_ONLY_LAT_SGL, SETUP_NONE, 0, 32);

    // vle8 (Unit)
    MEASURE("vle8.v (Unit) TP", VLE8_TP_BLK, VLE8_TP_SGL, SETUP_NONE, 31, 32);
    MEASURE("vle8.v (Unit) LAT", VLE8_LAT_BLK, VLE8_LAT_SGL, SETUP_NONE, 31, 32);
    
    // vlse8 (Stride)
    MEASURE("vlse8.v (Stride) TP", VLSE8_TP_BLK, VLSE8_TP_SGL, SETUP_NONE, 1984, 2048);
    MEASURE("vlse8.v (Stride) LAT", VLSE8_LAT_BLK, VLSE8_LAT_SGL, SETUP_NONE, 1984, 2048);

    // vluxei32 (Contig)
    MEASURE("vluxei32.v (Contig) TP", VLUXEI32_TP_BLK, VLUXEI32_TP_SGL, SETUP_CONTIG, 31, 32);
    MEASURE("vluxei32.v (Contig) LAT", VLUXEI32_LAT_BLK, VLUXEI32_LAT_SGL, SETUP_CONTIG, 31, 32);

    // vluxei32 (Stride)
    MEASURE("vluxei32.v (Stride) TP", VLUXEI32_TP_BLK, VLUXEI32_TP_SGL, SETUP_STRIDED, 1984, 2048);
    MEASURE("vluxei32.v (Stride) LAT", VLUXEI32_LAT_BLK, VLUXEI32_LAT_SGL, SETUP_STRIDED, 1984, 2048);

    // vloxei32 (Stride)
    MEASURE("vloxei32.v (Stride) TP", VLOXEI32_TP_BLK, VLOXEI32_TP_SGL, SETUP_STRIDED, 1984, 2048);
    MEASURE("vloxei32.v (Stride) LAT", VLOXEI32_LAT_BLK, VLOXEI32_LAT_SGL, SETUP_STRIDED, 1984, 2048);
}

int main() {
    init_perf();
    
    int8_t *A = NULL;
    if (posix_memalign((void**)&A, 64, 16 * 1024 * 1024) != 0) {
        return 1;
    }
    memset(A, 0, 16 * 1024 * 1024);

    uint32_t idx_contig[32];
    uint32_t idx_strided[32];

    for(int i=0; i<32; i++) {
        idx_contig[i] = i;
    }

    int lda = 64; // row stride
    for(int i=0; i<32; i++) {
        int r = i / 8;
        int c = i % 8;
        idx_strided[i] = r * lda + c;
    }

    printf("Starting Advanced Vector Load Benchmark (v3)...\n");
    
    // 4KB L1 Cache
    run_sweep(4 * 1024, A, idx_contig, idx_strided);
    
    // 512KB L2 Cache
    run_sweep(512 * 1024, A, idx_contig, idx_strided);
    
    // 16MB DRAM
    run_sweep(16 * 1024 * 1024, A, idx_contig, idx_strided);

    free(A);
    return 0;
}
