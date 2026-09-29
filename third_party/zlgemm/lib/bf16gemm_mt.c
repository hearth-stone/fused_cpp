// bf16gemm_mt.c — OpenMP Multi-threaded BF16 GEMM (library)
//
// Auto-selects M-tiling or N-tiling based on shape.
// B must be pre-packed by caller (bf16_pack_B) outside the timed region.
//
// Exported API:
//   void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N);
//   void bf16gemm_mt_dispatch(const bf16_t *A, const bf16_t *B_reo,
//                              f32_t *C, int M, int K_r, int N_r, int nthreads);
//   void bf16gemm_mt(const bf16_t *A, const bf16_t *B,
//                     f32_t *C, int M, int K, int N, int nthreads);
//
// Build:
//   cc -c bf16gemm_mt.c -march=armv8.6-a+bf16 -O2 -Wall -fopenmp
//   (link with bf16gemm_k.S and -fopenmp -lm)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "gemm_params.h"

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════
typedef uint16_t bf16_t;
typedef float    f32_t;

enum { BF16_NEON_N_TILE = 8 };

static size_t round_up_64(size_t size) {
    return (size + 63u) & ~(size_t)63u;
}

static void *aligned_alloc_64(size_t size) {
    if (size == 0)
        size = 64;
    return aligned_alloc(64, round_up_64(size));
}

static int bf16_effective_threads(int num_threads, int M, int n_tiles) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;

    const char *clamp_env = getenv("BF16_NEON_CLAMP_THREADS");
    int clamp_threads = clamp_env ? atoi(clamp_env) != 0 : 1;
    if (!clamp_threads)
        return num_threads;

    int work_units = ((M + 7) / 8) * n_tiles;
    if (work_units < 1)
        work_units = 1;
    if (num_threads > work_units)
        num_threads = work_units;

    if (M < 64 && n_tiles < num_threads * 4) {
        int by_n = (n_tiles + 3) / 4;
        if (by_n < 1)
            by_n = 1;
        if (num_threads > by_n)
            num_threads = by_n;
    }
    return num_threads;
}

static int bf16_use_n_split(int M, int K_r, int N_r, int n_tiles,
                            int num_threads) {
    const char *split = getenv("BF16_NEON_SPLIT");
    if (!split)
        split = getenv("BF16_SVE_SPLIT");
    if (split) {
        if (strcmp(split, "m") == 0)
            return 0;
        if (strcmp(split, "nblock") == 0)
            return 0;
        if (strcmp(split, "n") == 0)
            return n_tiles >= num_threads;
    }

    if (num_threads <= 1 || n_tiles < num_threads)
        return 0;

    /*
     * 80c opt-parts data shows high-thread NEON BF16 large-M cases often do
     * better with M splitting.  N splitting partitions B, but it can make each
     * thread sweep a thin M shard and exposes online A pack / C-store overhead.
     * Keep N splitting for very wide-N cases via the rule below.
     */
    if (num_threads >= 16 && M >= 512 && K_r >= 512 && N_r <= M * 4)
        return 0;
    if (num_threads >= 16 && M >= 128 && K_r >= 2048 && N_r <= 1024)
        return 0;

    if (M / 8 < num_threads)
        return 1;

    const size_t b_panel_bytes = (size_t)K_r * (size_t)N_r * sizeof(bf16_t);
    return b_panel_bytes >= 512u * 1024u && N_r >= M * 2;
}

static int bf16_use_nblock_split(int n_tiles) {
    const char *split = getenv("BF16_NEON_SPLIT");
    if (!split)
        split = getenv("BF16_SVE_SPLIT");
    return split && strcmp(split, "nblock") == 0 && n_tiles > 0;
}

#ifndef __APPLE__
static long bf16_read_sysfs_ll(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char buf[128];
    long v = -1;
    if (fgets(buf, sizeof(buf), f)) {
        char *end = NULL;
        v = strtol(buf, &end, 10);
        if (end && (*end == 'K' || *end == 'k'))
            v *= 1024;
        else if (end && (*end == 'M' || *end == 'm'))
            v *= 1024 * 1024;
        else if (end && (*end == 'G' || *end == 'g'))
            v *= 1024L * 1024 * 1024;
    }
    fclose(f);
    return v;
}

static int bf16_count_shared_cpus(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char buf[512];
    int total = 0;
    if (fgets(buf, sizeof(buf), f)) {
        char *p = buf;
        while (*p) {
            long a = strtol(p, &p, 10);
            if (*p == '-') {
                p++;
                long b = strtol(p, &p, 10);
                total += (int)(b - a + 1);
            } else {
                total += 1;
            }
            while (*p == ',')
                p++;
            if (*p == '\n' || *p == '\0')
                break;
        }
    }
    fclose(f);
    return total > 0 ? total : -1;
}
#endif

static void bf16_llc_facts(long *llc_bytes, int *llc_cores) {
    static long cached_bytes = -2;
    static int cached_cores = -2;
    if (cached_bytes != -2) {
        *llc_bytes = cached_bytes;
        *llc_cores = cached_cores;
        return;
    }

    long best_bytes = -1;
    int best_cores = -1;
#ifdef __APPLE__
    uint64_t apple_l2_u64 = 0;
    size_t sz = sizeof(apple_l2_u64);
    if (sysctlbyname("hw.perflevel0.l2cachesize", &apple_l2_u64, &sz,
                     NULL, 0) == 0) {
        best_bytes = (long)apple_l2_u64;
    } else {
        int apple_l2_i = -1;
        sz = sizeof(apple_l2_i);
        if (sysctlbyname("hw.perflevel0.l2cachesize", &apple_l2_i, &sz,
                         NULL, 0) == 0)
            best_bytes = apple_l2_i;
    }
    int apple_cores = -1;
    sz = sizeof(apple_cores);
    if (sysctlbyname("hw.perflevel0.cpusperl2", &apple_cores, &sz, NULL, 0) == 0)
        best_cores = apple_cores;
#else
    for (int idx = 0; idx < 8; idx++) {
        char sp[128], cp[128];
        snprintf(sp, sizeof(sp),
                 "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        snprintf(cp, sizeof(cp),
                 "/sys/devices/system/cpu/cpu0/cache/index%d/shared_cpu_list",
                 idx);
        long b = bf16_read_sysfs_ll(sp);
        if (b > best_bytes) {
            best_bytes = b;
            best_cores = bf16_count_shared_cpus(cp);
        }
    }
#endif
    cached_bytes = best_bytes;
    cached_cores = best_cores;
    *llc_bytes = best_bytes;
    *llc_cores = best_cores;
}

static int bf16_round_to_candidate(int x) {
    static const int cand[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int best = 1;
    for (unsigned k = 0; k < sizeof(cand) / sizeof(cand[0]); k++) {
        if (cand[k] <= x)
            best = cand[k];
    }
    return best;
}

static int bf16_tile_model_nb(int n_tiles, int K_r, int n_tile,
                              int num_threads) {
    const int elem = 2;
    long llc_bytes;
    int llc_cores;
    bf16_llc_facts(&llc_bytes, &llc_cores);
    if (llc_bytes <= 0)
        llc_bytes = 32L * 1024 * 1024;
    if (llc_cores <= 0)
        llc_cores = num_threads > 0 ? num_threads : 1;

    int concurrent = num_threads < llc_cores ? num_threads : llc_cores;
    if (concurrent < 1)
        concurrent = 1;

    long target = llc_bytes / concurrent;
    long per_tile = (long)K_r * n_tile * elem;
    int upper = per_tile > 0 ? (int)(target / per_tile) : n_tiles;
    int lower = 64 / n_tile;
    if (lower < 1)
        lower = 1;
    int par_max = num_threads > 0 ? n_tiles / num_threads : n_tiles;
    if (par_max < 1)
        par_max = 1;

    int nb = upper < par_max ? upper : par_max;
    if (nb < lower)
        nb = lower;
    nb = bf16_round_to_candidate(nb);
    if (nb > n_tiles)
        nb = n_tiles;
    if (nb < 1)
        nb = 1;
    return nb;
}

static int bf16_nblock_tiles(int n_tiles, int K_r, int num_threads) {
    int block = 0;
    const char *env = getenv("BF16_NEON_NB_TILES");
    if (!env)
        env = getenv("BF16_SVE_NB_TILES");
    if (env)
        block = atoi(env);
    if (block <= 0) {
        const char *model = getenv("BF16_NEON_TILE_MODEL");
        if (!model)
            model = getenv("BF16_SVE_TILE_MODEL");
        if (model && atoi(model) != 0)
            block = bf16_tile_model_nb(n_tiles, K_r, BF16_NEON_N_TILE,
                                       num_threads);
        else
            block = 8;
    }
    if (block > n_tiles)
        block = n_tiles;
    if (block < 1)
        block = 1;
    return block;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel declarations (from bf16gemm_k.S)
// ═══════════════════════════════════════════════════════════════════════
void bf16gemm_k_ld (const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, bf16_t *A_reorder,
                    const gemm_params_t *params);
void bf16gemm_k_ld1(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, bf16_t *A_reorder,
                    const gemm_params_t *params);
void bf16gemm_k_ld2(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, bf16_t *A_reorder,
                    const gemm_params_t *params);
void bf16gemm_k_ld4(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, bf16_t *A_reorder,
                    const gemm_params_t *params);

// ═══════════════════════════════════════════════════════════════════════
// bf16_pack_B — Pack B matrix for bfmmla kernel (caller allocates B_reo)
//
// B_reo size: K * N bf16 elements.
// Layout (outer→inner):
//   for each N-block (8 cols):
//     for each K-block (4 rows):
//       for each of 4 column-pairs: 4 K-rows of c0, then 4 K-rows of c1
//
// One N-block = K * 8 bf16 values = K * 16 bytes.
// ═══════════════════════════════════════════════════════════════════════
void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N / 8; ++cb) {
        for (int rb = 0; rb < K / 4; ++rb) {
            int row_base = rb * 4, col_base = cb * 8;
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = col_base + cp * 2, c1 = c0 + 1;
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c0];
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c1];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// bf16_dispatch — Single-thread kernel dispatch (internal)
//
// Handles arbitrary M by decomposing into full-8 + tail-4/2/1 calls.
// ldc_global: C row stride in f32 elements (may differ from n for N-tiling).
// ═══════════════════════════════════════════════════════════════════════
static void bf16_dispatch(const bf16_t *A, const bf16_t *B_reo,
                           f32_t *C, int m, int k, int n,
                           bf16_t *A_reorder, int ldc_global) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        bf16gemm_k_ld(A, B_reo, C, A_reorder, (const gemm_params_t *)&p);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const bf16_t *At = A + (uint64_t)processed * k;
    f32_t        *Ct = C + (uint64_t)processed * ldc_global;
    bf16_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        bf16gemm_k_ld4(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        bf16gemm_k_ld2(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        bf16gemm_k_ld1(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_M — M-tiling: split M rows across threads
// ═══════════════════════════════════════════════════════════════════════
static void tile_M(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int M, int K_r, int N_r,
                    bf16_t *A_reo_pool, int num_threads) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int blocks_per_thread = M8 / num_threads;
    int extra_blocks       = M8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        int my_m = my_blocks * 8 + ((tid == num_threads - 1) ? M_rem : 0);

        if (my_m > 0) {
            int my_m_start = start_block * 8;
            const bf16_t *my_A = A + (uint64_t)my_m_start * K_r;
            f32_t        *my_C = C + (uint64_t)my_m_start * N_r;
            bf16_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;

            bf16_dispatch(my_A, B_reo, my_C, my_m, K_r, N_r,
                          my_A_reo, /*ldc_global=*/N_r);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_N — N-tiling: split N columns across threads (for small M)
//
// Each thread allocates its own A_reorder because each thread's "first N"
// iteration for each M-block writes A_reorder independently.
// B is shared read-only; each thread offsets into its N-block range.
// ═══════════════════════════════════════════════════════════════════════
static void tile_N(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int M, int K_r, int N_r,
                    bf16_t *A_reo_pool, int num_threads) {
    int N8 = N_r / 8;
    int blocks_per_thread = N8 / num_threads;
    int extra_blocks       = N8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        if (my_blocks > 0) {
            int my_n       = my_blocks * 8;
            int my_n_start = start_block * 8;

            bf16_t *my_A_reo = A_reo_pool +
                (size_t)tid * (size_t)(M + 8) * (size_t)K_r;
            if (my_A_reo) {
                // B: each N-block = K_r * 8 bf16 values
                const bf16_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                // C: column offset; ldc_global = N_r (full row stride)
                f32_t *my_C = C + my_n_start;

                bf16_dispatch(A, my_B, my_C, M, K_r, my_n,
                              my_A_reo, /*ldc_global=*/N_r);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_NBlock — grouped-N split: parallelize coarse contiguous N groups.
//
// BF16_NEON_NB_TILES controls the group width in 8-column NEON tiles.  The
// BF16_SVE_* aliases are accepted so the SVE tile-model scripts can be reused
// with the NEON binary when validating the same B_group formula.
// ═══════════════════════════════════════════════════════════════════════
static void tile_NBlock(const bf16_t *A, const bf16_t *B_reo,
                        f32_t *C, int M, int K_r, int N_r,
                        bf16_t *A_reo_pool, int num_threads) {
    const int n_tiles = N_r / BF16_NEON_N_TILE;
    const int nb_tiles = bf16_nblock_tiles(n_tiles, K_r, num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        bf16_t *my_A_reo = A_reo_pool +
            (size_t)tid * (size_t)(M + 8) * (size_t)K_r;

        #pragma omp for schedule(static)
        for (int nb0 = 0; nb0 < n_tiles; nb0 += nb_tiles) {
            int nb_end = nb0 + nb_tiles;
            if (nb_end > n_tiles)
                nb_end = n_tiles;
            int n0 = nb0 * BF16_NEON_N_TILE;
            int nb_cols = (nb_end - nb0) * BF16_NEON_N_TILE;

            const bf16_t *my_B = B_reo +
                (size_t)nb0 * (size_t)K_r * (size_t)BF16_NEON_N_TILE;
            f32_t *my_C = C + n0;
            bf16_dispatch(A, my_B, my_C, M, K_r, nb_cols, my_A_reo, N_r);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// bf16gemm_mt_dispatch — Core multi-threaded compute (no allocation, no pack)
//
// A:   M × K_r  row-major bf16 (padded to K_r multiple of 8)
// B_reo: pre-packed by bf16_pack_B, K_r × N_r bf16
// C:   M × N_r  row-major f32 (output, zero-init before call)
// K_r, N_r: must be multiples of 8
//
// This is the function to put inside the timed benchmark region.
// ═══════════════════════════════════════════════════════════════════════
void bf16gemm_mt_dispatch(const bf16_t *A, const bf16_t *B_reo,
                           f32_t *C, int M, int K_r, int N_r,
                           int num_threads) {
    const int n_tiles = N_r / BF16_NEON_N_TILE;
    num_threads = bf16_effective_threads(num_threads, M, n_tiles);
    const int use_nblock_split = bf16_use_nblock_split(n_tiles);
    const int use_n_split = bf16_use_n_split(M, K_r, N_r, n_tiles, num_threads);

    if (use_nblock_split) {
        // Grouped-N: one scratch slice per thread because each N group repacks A.
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)num_threads * (size_t)(M + 8) * (size_t)K_r *
            sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_NBlock(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    } else if (!use_n_split) {
        // M-tiling: one shared A_reorder pool, partitioned by M rows.
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)(M + 8) * K_r * sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_M(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    } else {
        // N-tiling: fixed per-thread A_reorder slices avoid malloc/free in workers.
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)num_threads * (size_t)(M + 8) * (size_t)K_r *
            sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_N(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// bf16gemm_mt — Convenience wrapper: pad, pack B, then dispatch
//
// A_orig: M×K row-major bf16 (K need not be multiple of 8)
// B_orig: K×N row-major bf16 (K,N need not be multiples of 8)
// C:      M×N row-major f32 output (zero-init before call)
//
// Packs B internally — for repeated calls with the same B, prefer
// bf16_pack_B + bf16gemm_mt_dispatch to avoid re-packing.
// ═══════════════════════════════════════════════════════════════════════
void bf16gemm_mt(const bf16_t *A_orig, const bf16_t *B_orig,
                  f32_t *C, int M, int K, int N,
                  int num_threads) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;

    // Pack B
    bf16_t *B_reo = (bf16_t *)aligned_alloc_64(
        (size_t)K_r * N_r * sizeof(bf16_t));
    if (!B_reo) return;

    bf16_t *B_pad = NULL;
    const bf16_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (bf16_t *)calloc((size_t)K_r * N_r, sizeof(bf16_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(bf16_t));
        B_use = B_pad;
    }
    bf16_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    // Pad A if needed
    bf16_t *A_pad = NULL;
    const bf16_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (bf16_t *)calloc((size_t)M * K_r, sizeof(bf16_t));
        if (!A_pad) { free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(bf16_t));
        A_use = A_pad;
    }

    if (N_r == N) {
        bf16gemm_mt_dispatch(A_use, B_reo, C, M, K_r, N_r, num_threads);
    } else {
        f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
        if (!C_pad) { free(A_pad); free(B_reo); return; }
        bf16gemm_mt_dispatch(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(f32_t));
        free(C_pad);
    }

    free(A_pad);
    free(B_reo);
}

// ═══════════════════════════════════════════════════════════════════════
// BF16 GEMM bf16 output, no C load (_nld_b suffix) — zero-init accums
//
// C[i][j] = sum_k(A[i][k] * B[k][j])     (bf16 output)
//
// Accumulators start at zero (no C read). Since bfmmla accumulators are
// fp32, result is truncated to bf16 on store via bfcvtn.
// ═══════════════════════════════════════════════════════════════════════

// ── Kernel declarations (from bf16gemm_k.S) ───────────────────────────
void bf16gemm_k_nld_b (const bf16_t *A, const bf16_t *B_reo,
                        bf16_t *C, bf16_t *A_reorder,
                        const gemm_params_t *params);
void bf16gemm_k_nld1_b(const bf16_t *A, const bf16_t *B_reo,
                        bf16_t *C, bf16_t *A_reorder,
                        const gemm_params_t *params);
void bf16gemm_k_nld2_b(const bf16_t *A, const bf16_t *B_reo,
                        bf16_t *C, bf16_t *A_reorder,
                        const gemm_params_t *params);
void bf16gemm_k_nld4_b(const bf16_t *A, const bf16_t *B_reo,
                        bf16_t *C, bf16_t *A_reorder,
                        const gemm_params_t *params);

// ── bf16_nld_b_dispatch — Single-thread nld bf16 kernel dispatch ─────
static void bf16_nld_b_dispatch(const bf16_t *A, const bf16_t *B_reo,
                                 bf16_t *C, int m, int k, int n,
                                 bf16_t *A_reorder, int ldc_global) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        bf16gemm_k_nld_b(A, B_reo, C, A_reorder, (const gemm_params_t *)&p);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const bf16_t *At = A + (uint64_t)processed * k;
    bf16_t       *Ct = C + (uint64_t)processed * ldc_global;
    bf16_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        bf16gemm_k_nld4_b(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        bf16gemm_k_nld2_b(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        bf16gemm_k_nld1_b(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
    }
}

// ── tile_M_bf16_nld_b — M-tiling for nld bf16 output ─────────────────
static void tile_M_bf16_nld_b(const bf16_t *A, const bf16_t *B_reo,
                               bf16_t *C, int M, int K_r, int N_r,
                               bf16_t *A_reo_pool, int num_threads) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int blocks_per_thread = M8 / num_threads;
    int extra_blocks       = M8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }
        int my_m = my_blocks * 8 + ((tid == num_threads - 1) ? M_rem : 0);
        if (my_m > 0) {
            int my_m_start = start_block * 8;
            const bf16_t *my_A = A + (uint64_t)my_m_start * K_r;
            bf16_t       *my_C = C + (uint64_t)my_m_start * N_r;
            bf16_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;
            bf16_nld_b_dispatch(my_A, B_reo, my_C, my_m, K_r, N_r,
                                my_A_reo, /*ldc_global=*/N_r);
        }
    }
}

// ── tile_N_bf16_nld_b — N-tiling for nld bf16 output ─────────────────
static void tile_N_bf16_nld_b(const bf16_t *A, const bf16_t *B_reo,
                               bf16_t *C, int M, int K_r, int N_r,
                               bf16_t *A_reo_pool, int num_threads) {
    int N8 = N_r / 8;
    int blocks_per_thread = N8 / num_threads;
    int extra_blocks       = N8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }
        if (my_blocks > 0) {
            int my_n       = my_blocks * 8;
            int my_n_start = start_block * 8;
            bf16_t *my_A_reo = A_reo_pool +
                (size_t)tid * (size_t)(M + 8) * (size_t)K_r;
            if (my_A_reo) {
                const bf16_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                bf16_t       *my_C = C + my_n_start;
                bf16_nld_b_dispatch(A, my_B, my_C, M, K_r, my_n,
                                    my_A_reo, /*ldc_global=*/N_r);
            }
        }
    }
}

// ── bf16gemm_mt_dispatch_nld_b — Core multi-threaded nld bf16 compute ─
void bf16gemm_mt_dispatch_nld_b(const bf16_t *A, const bf16_t *B_reo,
                                 bf16_t *C, int M, int K_r, int N_r,
                                 int num_threads) {
    const int n_tiles = N_r / 8;
    num_threads = bf16_effective_threads(num_threads, M, n_tiles);
    const int use_n_split = bf16_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    if (!use_n_split) {
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)(M + 8) * K_r * sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_M_bf16_nld_b(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    } else {
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)num_threads * (size_t)(M + 8) * (size_t)K_r *
            sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_N_bf16_nld_b(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    }
}

// ── bf16gemm_mt_nld_b — Convenience wrapper for nld bf16 output ──────
void bf16gemm_mt_nld_b(const bf16_t *A_orig, const bf16_t *B_orig,
                        bf16_t *C, int M, int K, int N,
                        int num_threads) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;

    bf16_t *B_reo = (bf16_t *)aligned_alloc_64(
        (size_t)K_r * N_r * sizeof(bf16_t));
    if (!B_reo) return;

    bf16_t *B_pad = NULL;
    const bf16_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (bf16_t *)calloc((size_t)K_r * N_r, sizeof(bf16_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(bf16_t));
        B_use = B_pad;
    }
    bf16_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    bf16_t *A_pad = NULL;
    const bf16_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (bf16_t *)calloc((size_t)M * K_r, sizeof(bf16_t));
        if (!A_pad) { free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(bf16_t));
        A_use = A_pad;
    }

    if (N_r == N) {
        bf16gemm_mt_dispatch_nld_b(A_use, B_reo, C, M, K_r, N_r, num_threads);
    } else {
        bf16_t *C_pad = (bf16_t *)calloc((size_t)M * N_r, sizeof(bf16_t));
        if (!C_pad) { free(A_pad); free(B_reo); return; }
        bf16gemm_mt_dispatch_nld_b(A_use, B_reo, C_pad, M, K_r, N_r,
                                   num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(bf16_t));
        free(C_pad);
    }

    free(A_pad);
    free(B_reo);
}

// ═══════════════════════════════════════════════════════════════════════
// BF16 GEMM with fp32 bias (_bias_f suffix) — no C load, zero-init accums
//
// C[i][j] = sum_k(A[i][k] * B[k][j]) + bias[j]
//
// Accumulators start at zero (no C read). Bias is added in fp32 domain
// during STORE. Since bfmmla accumulators are already fp32, no conversion
// is needed — just zip → ldp bias → fadd → stp.
// ═══════════════════════════════════════════════════════════════════════

// ── Kernel declarations (from bf16gemm_k_bias.S) ──────────────────────
void bf16gemm_k_ld_bias_f (const bf16_t *A, const bf16_t *B_reo,
                            f32_t *C, bf16_t *A_reorder,
                            const gemm_params_t *params,
                            const f32_t *bias);
void bf16gemm_k_ld1_bias_f(const bf16_t *A, const bf16_t *B_reo,
                            f32_t *C, bf16_t *A_reorder,
                            const gemm_params_t *params,
                            const f32_t *bias);
void bf16gemm_k_ld2_bias_f(const bf16_t *A, const bf16_t *B_reo,
                            f32_t *C, bf16_t *A_reorder,
                            const gemm_params_t *params,
                            const f32_t *bias);
void bf16gemm_k_ld4_bias_f(const bf16_t *A, const bf16_t *B_reo,
                            f32_t *C, bf16_t *A_reorder,
                            const gemm_params_t *params,
                            const f32_t *bias);

// ── bf16_dispatch_bias_f — Single-thread bias kernel dispatch ─────────
static void bf16_dispatch_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                  f32_t *C, int m, int k, int n,
                                  bf16_t *A_reorder, int ldc_global,
                                  const f32_t *bias) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        bf16gemm_k_ld_bias_f(A, B_reo, C, A_reorder, (const gemm_params_t *)&p, bias);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const bf16_t *At = A + (uint64_t)processed * k;
    f32_t        *Ct = C + (uint64_t)processed * ldc_global;
    bf16_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        bf16gemm_k_ld4_bias_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p, bias);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        bf16gemm_k_ld2_bias_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p, bias);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        bf16gemm_k_ld1_bias_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p, bias);
    }
}

// ── tile_M_bf16_bias_f — M-tiling for bias fp32 output ────────────────
static void tile_M_bf16_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                f32_t *C, int M, int K_r, int N_r,
                                bf16_t *A_reo_pool, int num_threads,
                                const f32_t *bias) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int blocks_per_thread = M8 / num_threads;
    int extra_blocks       = M8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        int my_m = my_blocks * 8 + ((tid == num_threads - 1) ? M_rem : 0);

        if (my_m > 0) {
            int my_m_start = start_block * 8;
            const bf16_t *my_A = A + (uint64_t)my_m_start * K_r;
            f32_t        *my_C = C + (uint64_t)my_m_start * N_r;
            bf16_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;

            bf16_dispatch_bias_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                                 my_A_reo, /*ldc_global=*/N_r, bias);
        }
    }
}

// ── tile_N_bf16_bias_f — N-tiling for bias fp32 output ────────────────
static void tile_N_bf16_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                f32_t *C, int M, int K_r, int N_r,
                                bf16_t *A_reo_pool, int num_threads,
                                const f32_t *bias) {
    int N8 = N_r / 8;
    int blocks_per_thread = N8 / num_threads;
    int extra_blocks       = N8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        if (my_blocks > 0) {
            int my_n       = my_blocks * 8;
            int my_n_start = start_block * 8;

            bf16_t *my_A_reo = A_reo_pool +
                (size_t)tid * (size_t)(M + 8) * (size_t)K_r;
            if (my_A_reo) {
                const bf16_t *my_B    = B_reo + (uint64_t)start_block * K_r * 8;
                f32_t        *my_C    = C + my_n_start;
                const f32_t  *my_bias = bias + my_n_start;

                bf16_dispatch_bias_f(A, my_B, my_C, M, K_r, my_n,
                                     my_A_reo, /*ldc_global=*/N_r, my_bias);
            }
        }
    }
}

// ── bf16gemm_mt_dispatch_bias_f — Core multi-threaded bias compute ────
void bf16gemm_mt_dispatch_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                  f32_t *C, int M, int K_r, int N_r,
                                  int num_threads, const f32_t *bias) {
    const int n_tiles = N_r / 8;
    num_threads = bf16_effective_threads(num_threads, M, n_tiles);
    const int use_n_split = bf16_use_n_split(M, K_r, N_r, n_tiles, num_threads);

    if (!use_n_split) {
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)(M + 8) * K_r * sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_M_bf16_bias_f(A, B_reo, C, M, K_r, N_r, A_reo_pool,
                            num_threads, bias);
        free(A_reo_pool);
    } else {
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc_64(
            (size_t)num_threads * (size_t)(M + 8) * (size_t)K_r *
            sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_N_bf16_bias_f(A, B_reo, C, M, K_r, N_r, A_reo_pool,
                           num_threads, bias);
        free(A_reo_pool);
    }
}

// ── bf16gemm_mt_bias_f — Convenience wrapper ──────────────────────────
void bf16gemm_mt_bias_f(const bf16_t *A_orig, const bf16_t *B_orig,
                         f32_t *C, int M, int K, int N,
                         int num_threads, const f32_t *bias) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;

    // Pack B
    bf16_t *B_reo = (bf16_t *)aligned_alloc_64(
        (size_t)K_r * N_r * sizeof(bf16_t));
    if (!B_reo) return;

    bf16_t *B_pad = NULL;
    const bf16_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (bf16_t *)calloc((size_t)K_r * N_r, sizeof(bf16_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(bf16_t));
        B_use = B_pad;
    }
    bf16_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    // Pad bias if needed
    f32_t *bias_pad = NULL;
    const f32_t *bias_use;
    if (N_r == N) {
        bias_use = bias;
    } else {
        bias_pad = (f32_t *)calloc((size_t)N_r, sizeof(f32_t));
        if (!bias_pad) { free(B_reo); return; }
        memcpy(bias_pad, bias, N * sizeof(f32_t));
        bias_use = bias_pad;
    }

    // Pad A if needed
    bf16_t *A_pad = NULL;
    const bf16_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (bf16_t *)calloc((size_t)M * K_r, sizeof(bf16_t));
        if (!A_pad) { free(bias_pad); free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(bf16_t));
        A_use = A_pad;
    }

    if (N_r == N) {
        bf16gemm_mt_dispatch_bias_f(A_use, B_reo, C, M, K_r, N_r,
                                    num_threads, bias_use);
    } else {
        f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
        if (!C_pad) { free(A_pad); free(bias_pad); free(B_reo); return; }
        bf16gemm_mt_dispatch_bias_f(A_use, B_reo, C_pad, M, K_r, N_r,
                                    num_threads, bias_use);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(f32_t));
        free(C_pad);
    }

    free(A_pad);
    free(bias_pad);
    free(B_reo);
}
