#define _GNU_SOURCE

#include "bf16gemm.h"
#include "gemm_params.h"

#include <arm_sve.h>
#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void bf16gemm_k_nld_f_m12(const bf16_t *a, const bf16_t *b_reordered,
                          f32_t *c, bf16_t *a_reordered,
                          const gemm_params_t *params);
void bf16gemm_k_nld_f_nr_fullm(const bf16_t *a,
                               const bf16_t *b_reordered, f32_t *c,
                               bf16_t *a_reordered,
                               const gemm_params_t *params);

enum {
    DEFAULT_K = 2048,
    DEFAULT_RUNS = 51,
    DEFAULT_CPU = 48,
    M12_ROWS = 12,
    CACHE_LINE_BYTES = 64,
    COLD_TAIL_MIB = 192,
    EVICT_MIB = 4,
};

typedef enum {
    B_MODE_FRESH = 0,
    B_MODE_REUSE = 1,
    B_MODE_PREWARM = 2,
    B_MODE_EVICT = 3,
} b_mode_t;

static double now_sec(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1.0e-9;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static void *allocate_aligned(size_t bytes) {
    void *pointer = NULL;
    const size_t rounded_bytes = (bytes + 63u) & ~(size_t)63u;
    const int error = posix_memalign(&pointer, 64u, rounded_bytes);
    if (error != 0) {
        fprintf(stderr, "posix_memalign(%zu) failed: %s\n",
                rounded_bytes, strerror(error));
        exit(EXIT_FAILURE);
    }
    return pointer;
}

static int parse_int(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < 0 || value > INT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static void pin_cpu(int cpu) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET((size_t)cpu, &cpu_set);
    const int error = sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
    if (error != 0) {
        fprintf(stderr, "sched_setaffinity(cpu=%d) failed: %s\n",
                cpu, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static void initialize_bf16(bf16_t *data, size_t elements, uint32_t salt) {
    for (size_t index = 0; index < elements; ++index) {
        data[index] =
            (bf16_t)(0x3f00u + (((uint32_t)index * 5u + salt) & 15u));
    }
}

static uint64_t scan_cache_lines(const bf16_t *data, size_t elements) {
    volatile const bf16_t *volatile_data = data;
    const size_t line_elements = CACHE_LINE_BYTES / sizeof(*data);
    uint64_t checksum = 0;
    for (size_t index = 0; index < elements; index += line_elements)
        checksum += volatile_data[index];
    return checksum;
}

static uint64_t scan_bytes(const uint8_t *data, size_t bytes) {
    volatile const uint8_t *volatile_data = data;
    uint64_t checksum = 0;
    for (size_t index = 0; index < bytes; index += CACHE_LINE_BYTES)
        checksum += volatile_data[index];
    return checksum;
}

static double median(double *samples, int runs) {
    qsort(samples, (size_t)runs, sizeof(*samples), compare_double);
    return samples[runs / 2];
}

static void validate_common(int k_size, int n_size, int runs) {
    if (k_size <= 0 || (k_size % 8) != 0 ||
        n_size <= 0 || (n_size % 8) != 0 ||
        runs < 3 || (runs % 2) == 0) {
        fprintf(stderr,
                "K and N must be positive multiples of 8; runs must be odd "
                "and at least 3\n");
        exit(EXIT_FAILURE);
    }
}

static void run_fixed_mr(const bf16_t *a, const bf16_t *b, f32_t *c,
                         int m_size, int k_size, int n_size) {
    const int panels = m_size / M12_ROWS;
    const gemm_params_t params = {
        M12_ROWS, k_size, n_size, k_size, k_size, n_size,
    };
    for (int panel = 0; panel < panels; ++panel) {
        bf16_t *panel_a =
            (bf16_t *)a + (size_t)panel * M12_ROWS * (size_t)k_size;
        bf16gemm_k_nld_f_m12(
            panel_a, b,
            c + (size_t)panel * M12_ROWS * (size_t)n_size,
            panel_a, &params);
    }
}

static void run_fixed_nr(const bf16_t *a, const bf16_t *b, f32_t *c,
                         int m_size, int k_size, int n_size) {
    const int n_tile = (int)svcnth();
    const gemm_params_t params = {
        m_size, k_size, n_tile, k_size, k_size, n_size,
    };
    const size_t b_panel_elements =
        (size_t)k_size * (size_t)n_tile;
    for (int n_begin = 0; n_begin < n_size; n_begin += n_tile) {
        const size_t panel = (size_t)(n_begin / n_tile);
        bf16gemm_k_nld_f_nr_fullm(
            a, b + panel * b_panel_elements, c + n_begin,
            (bf16_t *)a, &params);
    }
}

static void run_n_group(const bf16_t *a, const bf16_t *b, f32_t *c,
                        int m_size, int k_size, int n_size,
                        int group_columns) {
    const int panels = m_size / M12_ROWS;
    for (int n_begin = 0; n_begin < n_size; n_begin += group_columns) {
        const int columns = n_size - n_begin < group_columns
            ? n_size - n_begin
            : group_columns;
        const gemm_params_t params = {
            M12_ROWS, k_size, columns, k_size, k_size, n_size,
        };
        const bf16_t *group_b =
            b + (size_t)n_begin * (size_t)k_size;
        for (int panel = 0; panel < panels; ++panel) {
            bf16_t *panel_a =
                (bf16_t *)a +
                (size_t)panel * M12_ROWS * (size_t)k_size;
            bf16gemm_k_nld_f_m12(
                panel_a, group_b,
                c + (size_t)panel * M12_ROWS * (size_t)n_size +
                    n_begin,
                panel_a, &params);
        }
    }
}

static void run_point(int m_size, int k_size, int n_size,
                      int runs, int cpu) {
    validate_common(k_size, n_size, runs);
    if (m_size <= 0 || (m_size % M12_ROWS) != 0) {
        fprintf(stderr, "M must be a positive multiple of 12\n");
        exit(EXIT_FAILURE);
    }
    pin_cpu(cpu);

    const int copies = runs + 5;
    const size_t a_elements = (size_t)m_size * (size_t)k_size;
    const size_t b_elements = (size_t)k_size * (size_t)n_size;
    const size_t c_elements = (size_t)m_size * (size_t)n_size;
    const size_t cold_bytes = (size_t)COLD_TAIL_MIB << 20;
    bf16_t *a = allocate_aligned(a_elements * sizeof(*a));
    bf16_t *b = allocate_aligned(
        (size_t)copies * b_elements * sizeof(*b));
    f32_t *c = allocate_aligned(c_elements * sizeof(*c));
    uint8_t *cold = allocate_aligned(cold_bytes);
    double *samples = allocate_aligned((size_t)runs * sizeof(*samples));

    initialize_bf16(a, a_elements, 1u);
    initialize_bf16(b, (size_t)copies * b_elements, 17u);
    memset(c, 0, c_elements * sizeof(*c));
    memset(cold, 1, cold_bytes);
    const uint64_t checksum = scan_bytes(cold, cold_bytes);

    for (int copy = 0; copy < copies; ++copy) {
        const bf16_t *copy_b = b + (size_t)copy * b_elements;
        const double start = now_sec();
        run_fixed_mr(a, copy_b, c, m_size, k_size, n_size);
        const double elapsed = now_sec() - start;
        if (copy >= 5)
            samples[copy - 5] = elapsed;
    }

    const double median_sec = median(samples, runs);
    const double operations =
        2.0 * (double)m_size * (double)k_size * (double)n_size;
    printf("mode=point M=%d K=%d N=%d runs=%d cpu=%d "
           "median_us=%.3f GFLOPS=%.2f checksum=%lu\n",
           m_size, k_size, n_size, runs, cpu, median_sec * 1.0e6,
           operations / median_sec / 1.0e9, (unsigned long)checksum);

    free(samples);
    free(cold);
    free(c);
    free(b);
    free(a);
}

static void run_compare(int m_size, int k_size, int n_size,
                        int runs, int cpu) {
    validate_common(k_size, n_size, runs);
    if (m_size <= 0 || (m_size % M12_ROWS) != 0) {
        fprintf(stderr, "M must be a positive multiple of 12\n");
        exit(EXIT_FAILURE);
    }
    pin_cpu(cpu);

    const int copies = 2 * (runs + 5) + 1;
    const size_t a_elements = (size_t)m_size * (size_t)k_size;
    const size_t b_elements = (size_t)k_size * (size_t)n_size;
    const size_t c_elements = (size_t)m_size * (size_t)n_size;
    const size_t cold_bytes = (size_t)COLD_TAIL_MIB << 20;
    bf16_t *a = allocate_aligned(a_elements * sizeof(*a));
    bf16_t *b = allocate_aligned(
        (size_t)copies * b_elements * sizeof(*b));
    f32_t *mr_output = allocate_aligned(c_elements * sizeof(*mr_output));
    f32_t *nr_output = allocate_aligned(c_elements * sizeof(*nr_output));
    uint8_t *cold = allocate_aligned(cold_bytes);
    double *mr_samples =
        allocate_aligned((size_t)runs * sizeof(*mr_samples));
    double *nr_samples =
        allocate_aligned((size_t)runs * sizeof(*nr_samples));

    initialize_bf16(a, a_elements, 7u);
    initialize_bf16(b, (size_t)copies * b_elements, 29u);
    memset(mr_output, 0, c_elements * sizeof(*mr_output));
    memset(nr_output, 0, c_elements * sizeof(*nr_output));
    memset(cold, 1, cold_bytes);

    run_fixed_mr(a, b, mr_output, m_size, k_size, n_size);
    run_fixed_nr(a, b, nr_output, m_size, k_size, n_size);
    if (memcmp(mr_output, nr_output,
               c_elements * sizeof(*mr_output)) != 0) {
        fprintf(stderr, "fixed-Mr and fixed-Nr outputs differ\n");
        exit(EXIT_FAILURE);
    }
    const uint64_t checksum = scan_bytes(cold, cold_bytes);

    int copy = 1;
    for (int round = 0; round < runs + 5; ++round) {
        const bf16_t *first_b = b + (size_t)copy++ * b_elements;
        const bf16_t *second_b = b + (size_t)copy++ * b_elements;
        double start;
        double mr_elapsed;
        double nr_elapsed;
        if ((round & 1) == 0) {
            start = now_sec();
            run_fixed_mr(a, first_b, mr_output,
                         m_size, k_size, n_size);
            mr_elapsed = now_sec() - start;
            start = now_sec();
            run_fixed_nr(a, second_b, nr_output,
                         m_size, k_size, n_size);
            nr_elapsed = now_sec() - start;
        } else {
            start = now_sec();
            run_fixed_nr(a, first_b, nr_output,
                         m_size, k_size, n_size);
            nr_elapsed = now_sec() - start;
            start = now_sec();
            run_fixed_mr(a, second_b, mr_output,
                         m_size, k_size, n_size);
            mr_elapsed = now_sec() - start;
        }
        if (round >= 5) {
            mr_samples[round - 5] = mr_elapsed;
            nr_samples[round - 5] = nr_elapsed;
        }
    }

    const double mr_sec = median(mr_samples, runs);
    const double nr_sec = median(nr_samples, runs);
    const double operations =
        2.0 * (double)m_size * (double)k_size * (double)n_size;
    printf("mode=compare M=%d K=%d N=%d runs=%d cpu=%d "
           "fixed_mr_us=%.3f fixed_nr_us=%.3f nr_gain_pct=%.2f "
           "fixed_mr_GFLOPS=%.2f fixed_nr_GFLOPS=%.2f exact=1 "
           "checksum=%lu\n",
           m_size, k_size, n_size, runs, cpu,
           mr_sec * 1.0e6, nr_sec * 1.0e6,
           (mr_sec / nr_sec - 1.0) * 100.0,
           operations / mr_sec / 1.0e9,
           operations / nr_sec / 1.0e9,
           (unsigned long)checksum);

    free(nr_samples);
    free(mr_samples);
    free(cold);
    free(nr_output);
    free(mr_output);
    free(b);
    free(a);
}

enum {
    GROUP_STRATEGIES = 4,
};

typedef enum {
    STRATEGY_FIXED_MR = 0,
    STRATEGY_FIXED_NR = 1,
    STRATEGY_GROUP_16 = 2,
    STRATEGY_GROUP_32 = 3,
} group_strategy_t;

static void run_group_strategy(group_strategy_t strategy,
                               const bf16_t *a, const bf16_t *b, f32_t *c,
                               int m_size, int k_size, int n_size) {
    switch (strategy) {
    case STRATEGY_FIXED_MR:
        run_fixed_mr(a, b, c, m_size, k_size, n_size);
        break;
    case STRATEGY_FIXED_NR:
        run_fixed_nr(a, b, c, m_size, k_size, n_size);
        break;
    case STRATEGY_GROUP_16:
        run_n_group(a, b, c, m_size, k_size, n_size, 16);
        break;
    case STRATEGY_GROUP_32:
        run_n_group(a, b, c, m_size, k_size, n_size, 32);
        break;
    }
}

static void run_compare_groups(int m_size, int k_size, int n_size,
                               int runs, int cpu) {
    validate_common(k_size, n_size, runs);
    if (m_size <= 0 || (m_size % M12_ROWS) != 0) {
        fprintf(stderr, "M must be a positive multiple of 12\n");
        exit(EXIT_FAILURE);
    }
    pin_cpu(cpu);

    const int copies = GROUP_STRATEGIES * (runs + 5) + 1;
    const size_t a_elements = (size_t)m_size * (size_t)k_size;
    const size_t b_elements = (size_t)k_size * (size_t)n_size;
    const size_t c_elements = (size_t)m_size * (size_t)n_size;
    const size_t cold_bytes = (size_t)COLD_TAIL_MIB << 20;
    bf16_t *a = allocate_aligned(a_elements * sizeof(*a));
    bf16_t *b = allocate_aligned(
        (size_t)copies * b_elements * sizeof(*b));
    f32_t *outputs[GROUP_STRATEGIES];
    double *samples[GROUP_STRATEGIES];
    for (int strategy = 0; strategy < GROUP_STRATEGIES; ++strategy) {
        outputs[strategy] =
            allocate_aligned(c_elements * sizeof(*outputs[strategy]));
        samples[strategy] =
            allocate_aligned((size_t)runs * sizeof(*samples[strategy]));
    }
    uint8_t *cold = allocate_aligned(cold_bytes);

    initialize_bf16(a, a_elements, 11u);
    initialize_bf16(b, (size_t)copies * b_elements, 31u);
    memset(cold, 1, cold_bytes);
    for (int strategy = 0; strategy < GROUP_STRATEGIES; ++strategy) {
        memset(outputs[strategy], 0,
               c_elements * sizeof(*outputs[strategy]));
        run_group_strategy(
            (group_strategy_t)strategy, a, b, outputs[strategy],
            m_size, k_size, n_size);
        if (strategy > 0 &&
            memcmp(outputs[0], outputs[strategy],
                   c_elements * sizeof(*outputs[strategy])) != 0) {
            fprintf(stderr, "group strategy %d output differs\n", strategy);
            exit(EXIT_FAILURE);
        }
    }
    const uint64_t checksum = scan_bytes(cold, cold_bytes);

    int copy = 1;
    for (int round = 0; round < runs + 5; ++round) {
        for (int position = 0; position < GROUP_STRATEGIES; ++position) {
            const group_strategy_t strategy =
                (group_strategy_t)((position + round) % GROUP_STRATEGIES);
            const bf16_t *copy_b = b + (size_t)copy++ * b_elements;
            const double start = now_sec();
            run_group_strategy(
                strategy, a, copy_b, outputs[strategy],
                m_size, k_size, n_size);
            const double elapsed = now_sec() - start;
            if (round >= 5)
                samples[strategy][round - 5] = elapsed;
        }
    }

    const double operations =
        2.0 * (double)m_size * (double)k_size * (double)n_size;
    double seconds[GROUP_STRATEGIES];
    for (int strategy = 0; strategy < GROUP_STRATEGIES; ++strategy)
        seconds[strategy] = median(samples[strategy], runs);
    printf("mode=groups M=%d K=%d N=%d runs=%d cpu=%d "
           "fixed_mr_us=%.3f fixed_nr_us=%.3f "
           "group16_us=%.3f group32_us=%.3f "
           "group16_gain_pct=%.2f group32_gain_pct=%.2f "
           "fixed_mr_GFLOPS=%.2f fixed_nr_GFLOPS=%.2f "
           "group16_GFLOPS=%.2f group32_GFLOPS=%.2f "
           "exact=1 checksum=%lu\n",
           m_size, k_size, n_size, runs, cpu,
           seconds[STRATEGY_FIXED_MR] * 1.0e6,
           seconds[STRATEGY_FIXED_NR] * 1.0e6,
           seconds[STRATEGY_GROUP_16] * 1.0e6,
           seconds[STRATEGY_GROUP_32] * 1.0e6,
           (seconds[STRATEGY_FIXED_MR] /
                seconds[STRATEGY_GROUP_16] -
            1.0) *
               100.0,
           (seconds[STRATEGY_FIXED_MR] /
                seconds[STRATEGY_GROUP_32] -
            1.0) *
               100.0,
           operations / seconds[STRATEGY_FIXED_MR] / 1.0e9,
           operations / seconds[STRATEGY_FIXED_NR] / 1.0e9,
           operations / seconds[STRATEGY_GROUP_16] / 1.0e9,
           operations / seconds[STRATEGY_GROUP_32] / 1.0e9,
           (unsigned long)checksum);

    free(cold);
    for (int strategy = GROUP_STRATEGIES - 1; strategy >= 0; --strategy) {
        free(samples[strategy]);
        free(outputs[strategy]);
    }
    free(b);
    free(a);
}

static void prepare_reused_b(b_mode_t mode, const bf16_t *reused_b,
                             size_t b_elements, const uint8_t *evict,
                             size_t evict_bytes, uint64_t *checksum) {
    if (mode == B_MODE_PREWARM)
        *checksum += scan_cache_lines(reused_b, b_elements);
    else if (mode == B_MODE_EVICT)
        *checksum += scan_bytes(evict, evict_bytes);
}

static void run_panels(int panels, int k_size, int n_size, int runs,
                       int cpu, b_mode_t mode) {
    validate_common(k_size, n_size, runs);
    if (panels < 2 || panels > 16 ||
        mode < B_MODE_FRESH || mode > B_MODE_EVICT) {
        fprintf(stderr, "panels must be in [2,16], B mode must be in [0,3]\n");
        exit(EXIT_FAILURE);
    }
    pin_cpu(cpu);

    const int copies = runs + 5;
    const size_t a_elements =
        (size_t)panels * M12_ROWS * (size_t)k_size;
    const size_t b_elements = (size_t)k_size * (size_t)n_size;
    const size_t c_elements =
        (size_t)panels * M12_ROWS * (size_t)n_size;
    const size_t cold_bytes = (size_t)COLD_TAIL_MIB << 20;
    const size_t evict_bytes = (size_t)EVICT_MIB << 20;
    const size_t b_copies_per_sample =
        mode == B_MODE_FRESH ? (size_t)panels : 1u;
    bf16_t *a = allocate_aligned(a_elements * sizeof(*a));
    bf16_t *b = allocate_aligned(
        (size_t)copies * b_copies_per_sample * b_elements * sizeof(*b));
    f32_t *c = allocate_aligned(c_elements * sizeof(*c));
    uint8_t *cold = allocate_aligned(cold_bytes);
    uint8_t *evict = allocate_aligned(evict_bytes);
    double *samples = allocate_aligned(
        (size_t)panels * (size_t)runs * sizeof(*samples));

    initialize_bf16(a, a_elements, 3u);
    initialize_bf16(
        b, (size_t)copies * b_copies_per_sample * b_elements, 19u);
    memset(c, 0, c_elements * sizeof(*c));
    memset(cold, 1, cold_bytes);
    memset(evict, 1, evict_bytes);
    uint64_t checksum = scan_bytes(cold, cold_bytes);
    const gemm_params_t params = {
        M12_ROWS, k_size, n_size, k_size, k_size, n_size,
    };

    for (int copy = 0; copy < copies; ++copy) {
        const bf16_t *base_b =
            b + (size_t)copy * b_copies_per_sample * b_elements;
        for (int panel = 0; panel < panels; ++panel) {
            const bf16_t *panel_b = mode == B_MODE_FRESH
                ? base_b + (size_t)panel * b_elements
                : base_b;
            if (panel > 0)
                prepare_reused_b(mode, base_b, b_elements, evict,
                                 evict_bytes, &checksum);
            bf16_t *panel_a =
                a + (size_t)panel * M12_ROWS * (size_t)k_size;
            const double start = now_sec();
            bf16gemm_k_nld_f_m12(
                panel_a, panel_b,
                c + (size_t)panel * M12_ROWS * (size_t)n_size,
                panel_a, &params);
            const double elapsed = now_sec() - start;
            if (copy >= 5) {
                samples[(size_t)panel * (size_t)runs +
                        (size_t)(copy - 5)] = elapsed;
            }
        }
    }

    printf("mode=panels panels=%d K=%d N=%d runs=%d cpu=%d "
           "b_mode=%d", panels, k_size, n_size, runs, cpu, (int)mode);
    for (int panel = 0; panel < panels; ++panel) {
        double *panel_samples =
            samples + (size_t)panel * (size_t)runs;
        printf(" p%d_us=%.3f", panel,
               median(panel_samples, runs) * 1.0e6);
    }
    printf(" checksum=%lu\n", (unsigned long)checksum);

    free(samples);
    free(evict);
    free(cold);
    free(c);
    free(b);
    free(a);
}

static void run_cache_state(int k_size, int n_size, int runs, int cpu) {
    validate_common(k_size, n_size, runs);
    pin_cpu(cpu);

    const int copies = runs + 5;
    const size_t a_elements = (size_t)M12_ROWS * (size_t)k_size;
    const size_t b_elements = (size_t)k_size * (size_t)n_size;
    const size_t c_elements = (size_t)M12_ROWS * (size_t)n_size;
    bf16_t *a = allocate_aligned(a_elements * sizeof(*a));
    bf16_t *b = allocate_aligned(
        3u * (size_t)copies * b_elements * sizeof(*b));
    f32_t *c = allocate_aligned(c_elements * sizeof(*c));
    double *cold = allocate_aligned((size_t)runs * sizeof(*cold));
    double *post_gemm =
        allocate_aligned((size_t)runs * sizeof(*post_gemm));
    double *hot = allocate_aligned((size_t)runs * sizeof(*hot));
    initialize_bf16(a, a_elements, 5u);
    initialize_bf16(
        b, 3u * (size_t)copies * b_elements, 23u);
    memset(c, 0, c_elements * sizeof(*c));

    const gemm_params_t params = {
        M12_ROWS, k_size, n_size, k_size, k_size, n_size,
    };
    uint64_t checksum = 0;
    for (int copy = 0; copy < copies; ++copy) {
        bf16_t *cold_b = b + (size_t)(3 * copy) * b_elements;
        bf16_t *gemm_b = cold_b + b_elements;
        bf16_t *hot_b = gemm_b + b_elements;

        double start = now_sec();
        checksum += scan_cache_lines(cold_b, b_elements);
        double end = now_sec();
        const double cold_elapsed = end - start;

        bf16gemm_k_nld_f_m12(a, gemm_b, c, a, &params);
        start = now_sec();
        checksum += scan_cache_lines(gemm_b, b_elements);
        end = now_sec();
        const double post_gemm_elapsed = end - start;

        checksum += scan_cache_lines(hot_b, b_elements);
        start = now_sec();
        checksum += scan_cache_lines(hot_b, b_elements);
        end = now_sec();
        const double hot_elapsed = end - start;

        if (copy >= 5) {
            const int sample = copy - 5;
            cold[sample] = cold_elapsed;
            post_gemm[sample] = post_gemm_elapsed;
            hot[sample] = hot_elapsed;
        }
    }

    const double bytes = (double)b_elements * sizeof(*b);
    const double cold_sec = median(cold, runs);
    const double post_gemm_sec = median(post_gemm, runs);
    const double hot_sec = median(hot, runs);
    printf("mode=cache K=%d N=%d B_KiB=%.0f runs=%d cpu=%d "
           "cold_us=%.3f post_gemm_us=%.3f hot_us=%.3f "
           "cold_GBps=%.2f post_gemm_GBps=%.2f hot_GBps=%.2f "
           "checksum=%lu\n",
           k_size, n_size, bytes / 1024.0, runs, cpu,
           cold_sec * 1.0e6, post_gemm_sec * 1.0e6,
           hot_sec * 1.0e6, bytes / cold_sec / 1.0e9,
           bytes / post_gemm_sec / 1.0e9, bytes / hot_sec / 1.0e9,
           (unsigned long)checksum);

    free(hot);
    free(post_gemm);
    free(cold);
    free(c);
    free(b);
    free(a);
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "usage:\n"
            "  %s point M K N [runs=%d] [cpu=%d]\n"
            "  %s compare M K N [runs=%d] [cpu=%d]\n"
            "  %s groups M K N [runs=%d] [cpu=%d]\n"
            "  %s panels PANELS K N B_MODE [runs=%d] [cpu=%d]\n"
            "  %s cache K N [runs=%d] [cpu=%d]\n"
            "B_MODE: 0=fresh per panel, 1=direct reuse, "
            "2=prewarm, 3=evict\n",
            program, DEFAULT_RUNS, DEFAULT_CPU,
            program, DEFAULT_RUNS, DEFAULT_CPU,
            program, DEFAULT_RUNS, DEFAULT_CPU,
            program, DEFAULT_RUNS, DEFAULT_CPU,
            program, DEFAULT_RUNS, DEFAULT_CPU);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "point") == 0 && argc >= 5) {
        run_point(parse_int(argv[2], "M"), parse_int(argv[3], "K"),
                  parse_int(argv[4], "N"),
                  argc > 5 ? parse_int(argv[5], "runs") : DEFAULT_RUNS,
                  argc > 6 ? parse_int(argv[6], "cpu") : DEFAULT_CPU);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "compare") == 0 && argc >= 5) {
        run_compare(
            parse_int(argv[2], "M"), parse_int(argv[3], "K"),
            parse_int(argv[4], "N"),
            argc > 5 ? parse_int(argv[5], "runs") : DEFAULT_RUNS,
            argc > 6 ? parse_int(argv[6], "cpu") : DEFAULT_CPU);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "groups") == 0 && argc >= 5) {
        run_compare_groups(
            parse_int(argv[2], "M"), parse_int(argv[3], "K"),
            parse_int(argv[4], "N"),
            argc > 5 ? parse_int(argv[5], "runs") : DEFAULT_RUNS,
            argc > 6 ? parse_int(argv[6], "cpu") : DEFAULT_CPU);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "panels") == 0 && argc >= 6) {
        run_panels(
            parse_int(argv[2], "panels"), parse_int(argv[3], "K"),
            parse_int(argv[4], "N"), argc > 6
                ? parse_int(argv[6], "runs") : DEFAULT_RUNS,
            argc > 7 ? parse_int(argv[7], "cpu") : DEFAULT_CPU,
            (b_mode_t)parse_int(argv[5], "B mode"));
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "cache") == 0 && argc >= 4) {
        run_cache_state(
            parse_int(argv[2], "K"), parse_int(argv[3], "N"),
            argc > 4 ? parse_int(argv[4], "runs") : DEFAULT_RUNS,
            argc > 5 ? parse_int(argv[5], "cpu") : DEFAULT_CPU);
        return EXIT_SUCCESS;
    }
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
