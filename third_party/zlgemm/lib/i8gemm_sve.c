// i8gemm_sve.c -- SVE I8 GEMM implementation with the same public API.
//
// Link this file instead of i8gemm_mt.c/i8gemm_k*.S when building the SVE
// variant. The exported function names intentionally match the NEON path.

#include <arm_sve.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_params.h"

typedef int8_t i8_t;
typedef int32_t i32_t;
typedef float f32_t;

static size_t round_up_64(size_t size) {
    return (size + 63u) & ~(size_t)63u;
}

static void *aligned_alloc_64(size_t size) {
    if (size == 0)
        size = 64;
    return aligned_alloc(64, round_up_64(size));
}

// Per-thread cached scratch for the A-reorder pools. The public dispatch packs
// A on every call; using a thread-local growing buffer removes the per-call
// aligned_alloc/free that dominated small-shape latency. At most two pools are
// live at once (primary = m12 or main, plus the m12 tail), so two slots suffice.
// Buffers are owned by the calling (master) thread; the OpenMP workers only
// write disjoint regions of an already-captured pointer, so this is race-free.
static __thread i8_t *g_a_poolA = NULL;
static __thread size_t g_a_capA = 0;
static __thread i8_t *g_a_poolB = NULL;
static __thread size_t g_a_capB = 0;

static i8_t *i8_scratch(i8_t **slot, size_t *cap, size_t n) {
    if (n == 0)
        return *slot;
    if (n > *cap) {
        free(*slot);
        *slot = (i8_t *)aligned_alloc_64(n);
        *cap = *slot ? round_up_64(n) : 0;
    }
    return *slot;
}

static int i8_sve_segments(void) {
    return (int)(svcntb() / 16);
}

static int i8_sve_n_tile(void) {
    return i8_sve_segments() * 8;
}

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

/*
 * Choose a 2D thread grid pm x pn over an (mblk x ntiles) tile grid. Each
 * thread owns an M-band and an N-band, so operand A is re-read by pn threads
 * and B by pm threads: aggregate traffic ~ pn*|A| + pm*|B| = K*(pn*M + pm*N).
 *
 * We do NOT require pm*pn == P or even division: that constraint makes the
 * grid unusable for non-power-of-2 thread counts (P=79 is prime, P=80=16*5)
 * exactly when a machine has many cores, which is when it matters most. With
 * the old even-only picker, a shape like 64x4096x1024 on 79/80 threads found
 * no grid and fell back to an M-split over ~6 row-blocks -- 70+ idle threads
 * and a double fork-join -- collapsing to ~1/9 of its t64 throughput.
 *
 * Instead, allow uneven bands (the caller distributes the remainder) and
 * pm*pn <= P. Cap pm <= mblk and pn <= ntiles so we never create empty bands.
 * Among candidates, maximise utilisation (pm*pn, the number of busy threads)
 * first, then minimise traffic pn*Mw + pm*Nw (puts more threads on the
 * dimension whose operand is cheaper to replicate). Returns pm=pn=1 in the
 * degenerate case; callers act on the grid only when pn>1 (a real 2D split).
 * Mw/Nw are the M and N extents (any consistent unit).
 */
static void i8_pick_grid(int mblk, int ntiles, int P, long Mw, long Nw,
                         int *out_pm, int *out_pn) {
    int best_pm = 1, best_pn = 1;
    long best_use = 0;
    double best_cost = 1e300;
    int pm_max = P < mblk ? P : mblk;
    for (int pm = 1; pm <= pm_max; pm++) {
        int pn = P / pm;
        if (pn > ntiles)
            pn = ntiles;
        if (pn < 1)
            pn = 1;
        long use = (long)pm * pn;
        double cost = (double)pn * (double)Mw + (double)pm * (double)Nw;
        if (use > best_use || (use == best_use && cost < best_cost)) {
            best_use = use;
            best_cost = cost;
            best_pm = pm;
            best_pn = pn;
        }
    }
    // 1D-grid rescue. When P is prime/awkward, the max-busy grid above is 1D
    // (pm==1 or pn==1), which re-streams one operand P times -> aggregate
    // bandwidth collapse (measured on Huawei: 2048x4096x2048 @t79 fell to 42%
    // of the compute ceiling because 79x1 re-reads B 79x). If a genuine 2D grid
    // exists that keeps >=7/8 of the threads busy, switch to the lowest-traffic
    // one. Factorable P (8/64/80 -> already 2D) is untouched: zero regression.
    if ((best_pm == 1 || best_pn == 1) && P >= 4) {
        long min_use = (long)P - P / 8;
        int r_pm = 0, r_pn = 0;
        double r_cost = 1e300;
        long r_use = 0;
        for (int pm = 2; pm <= pm_max; pm++) {
            int pn = P / pm;
            if (pn > ntiles) pn = ntiles;
            if (pn < 2) continue;                 // require a real 2D grid
            long use = (long)pm * pn;
            if (use < min_use) continue;
            double cost = (double)pn * (double)Mw + (double)pm * (double)Nw;
            if (cost < r_cost || (cost == r_cost && use > r_use)) {
                r_cost = cost; r_use = use; r_pm = pm; r_pn = pn;
            }
        }
        if (r_pn) { best_pm = r_pm; best_pn = r_pn; }
    }
    *out_pm = best_pm;
    *out_pn = best_pn;
}

static int i8_sve_effective_threads(int num_threads, int M, int K_r,
                                    int n_tiles, long min_macs_override) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;

    const char *clamp_env = getenv("I8_SVE_CLAMP_THREADS");
    int clamp_threads = clamp_env ? atoi(clamp_env) != 0 : 1;
    if (!clamp_threads)
        return num_threads;

    int work_units = ((M + 7) / 8) * n_tiles;
    if (work_units < 1)
        work_units = 1;
    if (num_threads > work_units)
        num_threads = work_units;

    /*
     * Fork-amortisation clamp. A static OpenMP region only pays off if each
     * worker does enough compute to cover the fork/barrier latency. Cap the
     * thread count so every thread gets at least I8_SVE_MIN_MACS multiply-adds.
     * The 512Ki default is calibrated so it NEVER reduces the small shapes the
     * 8-core dev machine already runs at full width (the smallest benchmarked
     * shapes, 64x128x512 = 4.2M macs, still allow >=8 threads), while it trims
     * the gross oversubscription that collapses the same shapes at 64/80
     * threads -- e.g. 64x512x256 (8.4M macs) caps at 16, matching its measured
     * t16 peak on the 80-core machine. Set I8_SVE_MIN_MACS=0 to disable.
     *
     * min_macs_override (>=0) lets the caller relax this for the hybrid no-pack
     * path, whose fork is a single cheap region (no separate A-pack team): a
     * small shape like 8x512x512 (2.1M macs) is capped to 4 threads by the
     * 512Ki default yet scales cleanly to 8 with the hybrid kernel. <0 uses the
     * env / 512Ki default (packed path -- unchanged, protects the 80-core box).
     */
    const char *mm_env = getenv("I8_SVE_MIN_MACS");
    long min_macs = mm_env ? atol(mm_env)
                           : (min_macs_override >= 0 ? min_macs_override
                                                     : (512L << 10));
    if (min_macs > 0) {
        long total_macs = (long)M * (long)K_r *
                          (long)(n_tiles * i8_sve_n_tile());
        int mac_cap = (int)(total_macs / min_macs);
        if (mac_cap < 1)
            mac_cap = 1;
        if (num_threads > mac_cap)
            num_threads = mac_cap;
    }

    if (M < 64 && n_tiles < num_threads * 4) {
        int by_n = (n_tiles + 3) / 4;
        if (by_n < 1)
            by_n = 1;
        if (num_threads > by_n)
            num_threads = by_n;
    }
    return num_threads;
}

/*
 * Advisory thread recommendation (the "knee"): the fewest threads that still
 * reach near-peak GOPS for this shape. Parallelism is bounded by the work-unit
 * count (mblk * n_tiles). Beyond that, extra threads help only until the shape
 * saturates memory bandwidth -- which happens early for a LARGE B panel but not
 * for a small (cache-resident) B that keeps scaling on N. So we only apply the
 * per-thread MAC floor (I8_REC_MACS, default 512Ki) when the B panel
 * (K_r*N_r bytes) exceeds I8_REC_CACHE (default 2Mi); small-B shapes scale on
 * work-units. This B-gated model was fitted to the Huawei 247-shape fine-sweep
 * knees: ~97% of shapes get a count >= their measured knee (no GOPS loss),
 * while bandwidth-bound large-B shapes are trimmed. NOTE: it is still
 * conservative -- it does NOT catch bandwidth-saturated *medium* shapes (whose
 * knee is set by B-streaming rate, not total MACs); exact knees need a per-
 * machine sweep (tests/huawei_fine_sweep.sh) to tune I8_REC_MACS/I8_REC_CACHE.
 * ADVISORY/opt-in only: i8gemm_mt_dispatch uses it solely when num_threads<=0
 * AND I8_SVE_AUTO_THREADS=1; the validated explicit-thread path is untouched.
 */
int i8gemm_recommend_threads(int M, int K_r, int N_r, int max_threads) {
    if (max_threads <= 0)
        max_threads = omp_get_max_threads();
    if (max_threads <= 0)
        max_threads = 1;
    int n_tile = i8_sve_n_tile();
    int n_tiles = N_r / n_tile;
    if (n_tiles < 1)
        n_tiles = 1;
    int mblk = (M + 7) / 8;
    if (mblk < 1)
        mblk = 1;
    long units = (long)mblk * (long)n_tiles;     /* upper bound on parallelism */
    long t = max_threads;
    if (t > units)
        t = units;
    const char *fe = getenv("I8_REC_MACS");
    long floor = fe ? atol(fe) : (512L << 10);    /* per-thread MAC floor */
    const char *ce = getenv("I8_REC_CACHE");
    long cache = ce ? atol(ce) : (2L << 20);      /* B-panel cache threshold */
    long b_bytes = (long)K_r * (long)N_r;
    if (floor > 0 && b_bytes > cache) {           /* bandwidth-bound: trim */
        long macs = (long)M * (long)K_r * (long)N_r;
        long cap = macs / floor;
        if (cap < 1)
            cap = 1;
        if (t > cap)
            t = cap;
    }
    if (t < 1)
        t = 1;
    return (int)t;
}

static int i8_sve_use_n_split(int M, int K_r, int N_r, int n_tiles,
                              int num_threads) {
    const char *split = getenv("I8_SVE_SPLIT");
    if (split) {
        if (strcmp(split, "m") == 0)
            return 0;
        if (strcmp(split, "n") == 0)
            return n_tiles >= num_threads;
    }

    if (num_threads <= 1) {
        if (M >= 64 && n_tiles > 1 &&
            (K_r >= 1024 || M >= 128 || N_r >= 2048))
            return 1;
        return 0;
    }

    if (n_tiles < num_threads)
        return 0;
    if (M / 8 < num_threads)
        return 1;

    /*
     * ACL's I8 selector often moves medium shapes away from a pure M-split
     * interleaved path.  On this platform, SVE I8 shapes around K,N >= 512
     * and M <= 512 benefit from N-split even when the packed-B panel is below
     * the old 512 KiB threshold: the A reorder is reused across N tiles and
     * the work distribution is much steadier across 4/8 cores.
     */
    if (M <= 512 && K_r >= 512 && N_r >= 512)
        return 1;

    const size_t b_panel_bytes = (size_t)K_r * (size_t)N_r * sizeof(i8_t);
    return b_panel_bytes >= 512u * 1024u && N_r >= M * 2;
}

void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N) {
    const int segs = i8_sve_segments();
    const int n_tile = segs * 8;
    int idx = 0;

    for (int nb = 0; nb < N; nb += n_tile) {
        for (int rb = 0; rb < K / 8; rb++) {
            int row_base = rb * 8;
            for (int cp = 0; cp < 4; cp++) {
                for (int sg = 0; sg < segs; sg++) {
                    int col_base = nb + sg * 8 + cp * 2;
                    for (int j = 0; j < 2; j++)
                        for (int i = 0; i < 8; i++)
                            B_reo[idx++] = B[(row_base + i) * N + col_base + j];
                }
            }
        }
    }
}


void i8gemm_k_ld(const i8_t *A, const i8_t *B_reo, i32_t *C,
                 i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_nld(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_nld_m12(const i8_t *A, const i8_t *B_reo, i32_t *C,
                      i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld1(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_nld1(const i8_t *A, const i8_t *B_reo, i32_t *C,
                   i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld2(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_nld2(const i8_t *A, const i8_t *B_reo, i32_t *C,
                   i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld4(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_nld4(const i8_t *A, const i8_t *B_reo, i32_t *C,
                   i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                   i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld1_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld2_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld4_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_scaled_f(const i8_t *A, const i8_t *B_reo, void *C,
                       i8_t *A_reorder, const gemm_params_t *params,
                       const f32_t *row_scale, const f32_t *column_scale,
                       const f32_t *bias);
void i8gemm_k_scaled_b(const i8_t *A, const i8_t *B_reo, void *C,
                       i8_t *A_reorder, const gemm_params_t *params,
                       const f32_t *row_scale, const f32_t *column_scale,
                       const f32_t *bias);
void i8gemm_k_scaled_m12_f(const i8_t *A, const i8_t *B_reo, void *C,
                           i8_t *A_reorder, const gemm_params_t *params,
                           const f32_t *row_scale,
                           const f32_t *column_scale, const f32_t *bias);
void i8gemm_k_scaled_m12_b(const i8_t *A, const i8_t *B_reo, void *C,
                           i8_t *A_reorder, const gemm_params_t *params,
                           const f32_t *row_scale,
                           const f32_t *column_scale, const f32_t *bias);
void i8gemm_k_ld_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                        i8_t *A_reorder, const gemm_params_t *params,
                        const f32_t *bias);
void i8gemm_k_ld1_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias);
void i8gemm_k_ld2_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias);
void i8gemm_k_ld4_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias);
void i8_pack_A_neon_m8_asm(const i8_t *A, i8_t *P, int K_r, int lda);

// Small-shape hybrid kernel: reads A directly (no reorder pass), contiguous
// deinterleave store. Defined in i8gemm_hybrid.S.
void i8gemm_k_hybrid(const i8_t *A, const i8_t *B_reo, i32_t *C,
                     i8_t *unused, const gemm_params_t *params);

// Narrow-M kernel (M in {1,2}): one row-pair, 4 smmla/K8 (no 8-row tile
// waste). Same B_reo layout as i8gemm_k_hybrid. Defined in i8gemm_hybrid.S.
void i8gemm_k_narrow(const i8_t *A, const i8_t *B_reo, i32_t *C,
                     i8_t *unused, const gemm_params_t *params);

// Narrow-M kernel for M in {3,4}: two row-pairs, 8 smmla/K8. Same B_reo layout.
void i8gemm_k_narrow2(const i8_t *A, const i8_t *B_reo, i32_t *C,
                      i8_t *unused, const gemm_params_t *params);

// Route to the hybrid (no-pack) kernel for tiny shapes where the A-reorder
// pass + per-tile overhead dominate. Crossover measured on Neoverse-V1: the
// hybrid wins for small M and small/medium K; the packed path wins once K or M
// grows enough to amortize the pack and exploit contiguous packed A loads.
static int i8_use_hybrid_for_shape(int M, int K_r, int N_r, int num_threads) {
    const char *env = getenv("I8_SVE_HYBRID");
    if (env)
        return atoi(env) != 0;
    if (num_threads >= 2) {
        // Multi-thread: the packed path pays a separate A-reorder OpenMP region
        // whose fork overhead dominates at small per-thread work, so the hybrid
        // (no pack pass, clean M-split / N-split) scales much better for any
        // shape with a small K and M. Large K/M keep the packed path (better
        // B reuse and contiguous packed-A loads amortize there). N is
        // unrestricted here: large-N small-M/K shapes benefit most.
        if (K_r <= 256 && M <= 256)
            return 1;
        // Decode/GEMV (M<=4): memory-bandwidth bound. The packed path's 512Ki
        // fork clamp starves it (1x4096x1024 pinned to ~8 threads = 266 GOPS on
        // 80c), while the hybrid path (direct-A read, lenient 64Ki clamp) scales
        // to the bandwidth knee. Measured on Huawei: hybrid 1.2-1.6x the packed
        // default for M=1 across K/N (1x4096x1024 266->437, 1x4096x4096 780->1164).
        // smmla wastes 7/8 of the tile at M=1 but that is hidden under the memory
        // wall, so a dedicated GEMV kernel is not worth it -- routing is the lever.
        if (M <= 4)
            return 1;
        // Small-M, larger-K: with at most two 8-row blocks (M<=16) a pure
        // M-split cannot fill the threads and the packed path additionally pays
        // its separate A-pack fork, while the hybrid path N-splits the
        // (L1-resident) B panel across all threads with one cheap fork.
        // Measured on V1: 8x512x512 @t8 624 -> ~1344 (vs ACL 1147), 16x512x512
        // @t8 1118 -> 1571. M>=24 is left to the packed path: its 2D grid +
        // m12 kernel (24 = 2x12) fill the threads and win there.
        if (M <= 16 && K_r <= 1024)
            return 1;
        return 0;
    }
    // Single thread: hybrid only helps the tiniest shapes (no pack pass to
    // amortize); for larger N the packed path's contiguous A wins.
    if (N_r > 256)
        return 0;
    if (K_r <= 128 && M <= 64)
        return 1;
    if (M <= 16 && K_r <= 256)
        return 1;
    return 0;
}

static size_t i8_a_reorder_stride(int K_r) {
    return (size_t)K_r * 8u;
}

static size_t i8_a_reorder_stride_m12(int K_r) {
    return (size_t)K_r * 12u;
}

static int i8_use_a_reorder_for_shape(int K_r, int n_tiles) {
    const char *env = getenv("I8_SVE_NO_A_REORDER");
    if (env && atoi(env) != 0)
        return 0;
    return K_r >= 64 && n_tiles > 1;
}

static int i8_use_m12_for_shape(int M, int K_r, int n_tiles) {
    const char *env = getenv("I8_SVE_NO_M12");
    if (env && atoi(env) != 0)
        return 0;
    return M >= 12 && K_r >= 64 && n_tiles > 1;
}

// Pick how many 12-row (m12) blocks to use, leaving the remainder to the
// 8-row kernel. The m12 kernel (12x2VL, 24 accumulators) has a better
// load:compute ratio than the 8x2VL kernel, but partial tail blocks always
// run a full 8-row pass, so an M like 16 wastes half of a 4-row tail.
//
// Cost model (single full kernel pass, compute-bound):
//   time(b) ~= b * 12 / eff12 + ceil((M - 12b) / 8) * 8 / eff8
// with measured eff12 ~= 0.83 and eff8 ~= 0.70 on Neoverse-V1. Scaling by
// eff12*eff8 gives the integer comparison cost below. We minimise over the
// number of m12 blocks b, which lets the selector fall back to pure 8-row
// blocking (b=0) for shapes such as M=13..16 and choose mixed blocking such
// as 12+8+8 for M=28, both of which beat the old "always floor(M/12)" rule.
static int i8_m12_block_count(int M) {
    const char *env = getenv("I8_SVE_M12_BLOCKS");
    if (env) {
        int v = atoi(env);
        if (v < 0)
            v = 0;
        if (v > M / 12)
            v = M / 12;
        return v;
    }

    const int eff12 = 83;   // m12 (12x2VL) full-pass efficiency, percent
    const int eff8 = 70;    // 8x2VL full-pass efficiency, percent
    const int max_b = M / 12;
    int best_b = 0;
    long best_cost = -1;
    for (int b = 0; b <= max_b; b++) {
        int tail = M - 12 * b;
        int tail_blocks = (tail + 7) / 8;
        long cost = (long)b * 12 * eff8 + (long)tail_blocks * 8 * eff12;
        if (best_cost < 0 || cost < best_cost) {
            best_cost = cost;
            best_b = b;
        }
    }
    return best_b;
}

static void i8_pack_A_sve_block(const i8_t *A, i8_t *A_reo,
                                int M, int K_r) {
    if (M == 8) {
        i8_pack_A_neon_m8_asm(A, A_reo, K_r, K_r);
        return;
    }

    size_t idx = 0;
    for (int kb = 0; kb < K_r; kb += 8) {
        for (int rp = 0; rp < 4; rp++) {
            int r0 = rp * 2;
            int r1 = r0 + 1;
            if (r0 < M)
                memcpy(A_reo + idx, A + (size_t)r0 * K_r + kb, 8);
            else
                memset(A_reo + idx, 0, 8);
            idx += 8;
            if (r1 < M)
                memcpy(A_reo + idx, A + (size_t)r1 * K_r + kb, 8);
            else
                memset(A_reo + idx, 0, 8);
            idx += 8;
        }
    }
}

static void i8_pack_A_sve_m12_block(const i8_t *A, i8_t *A_reo,
                                    int K_r) {
    size_t idx = 0;
    for (int kb = 0; kb < K_r; kb += 8) {
        for (int rp = 0; rp < 6; rp++) {
            int r0 = rp * 2;
            int r1 = r0 + 1;
            memcpy(A_reo + idx, A + (size_t)r0 * K_r + kb, 8);
            idx += 8;
            memcpy(A_reo + idx, A + (size_t)r1 * K_r + kb, 8);
            idx += 8;
        }
    }
}

static i8_t *i8_prepare_A_reorder_pool(const i8_t *A, int M, int K_r,
                                       int num_threads, i8_t *pool) {
    const int blocks = (M + 7) / 8;
    const size_t stride = i8_a_reorder_stride(K_r);
    if (!pool)
        return NULL;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int b = 0; b < blocks; b++) {
        int m0 = b * 8;
        int mb = M - m0 < 8 ? M - m0 : 8;
        i8_pack_A_sve_block(A + (size_t)m0 * K_r,
                            pool + (size_t)b * stride, mb, K_r);
    }
    return pool;
}

static i8_t *i8_prepare_A_reorder_pool_m12(const i8_t *A, int M, int K_r,
                                           int num_threads, i8_t *pool) {
    const int blocks = M / 12;
    const size_t stride = i8_a_reorder_stride_m12(K_r);
    if (!pool)
        return NULL;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int b = 0; b < blocks; b++) {
        i8_pack_A_sve_m12_block(A + (size_t)b * 12u * K_r,
                                pool + (size_t)b * stride, K_r);
    }
    return pool;
}

static void i8_dispatch_i32(const i8_t *A, const i8_t *B_reo, i32_t *C,
                            int M, int K_r, int N_r, int ldc,
                            int no_load, i8_t *A_reorder) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};

    if (no_load) {
        if (M <= 1)
            i8gemm_k_nld1(A, B_reo, C, A_reorder, &p);
        else if (M <= 2)
            i8gemm_k_nld2(A, B_reo, C, A_reorder, &p);
        else if (M <= 4)
            i8gemm_k_nld4(A, B_reo, C, A_reorder, &p);
        else
            i8gemm_k_nld(A, B_reo, C, A_reorder, &p);
    } else {
        if (M <= 1)
            i8gemm_k_ld1(A, B_reo, C, A_reorder, &p);
        else if (M <= 2)
            i8gemm_k_ld2(A, B_reo, C, A_reorder, &p);
        else if (M <= 4)
            i8gemm_k_ld4(A, B_reo, C, A_reorder, &p);
        else
            i8gemm_k_ld(A, B_reo, C, A_reorder, &p);
    }
}

static void i8_dispatch_i32_m12(const i8_t *A, const i8_t *B_reo, i32_t *C,
                                int K_r, int N_r, int ldc,
                                i8_t *A_reorder) {
    gemm_params_t p = {12, K_r, N_r, K_r, K_r, ldc};
    i8gemm_k_nld_m12(A, B_reo, C, A_reorder, &p);
}

static void i8_dispatch_f32(const i8_t *A, const i8_t *B_reo, f32_t *C,
                            int M, int K_r, int N_r, int ldc,
                            const f32_t *bias, i8_t *A_reorder) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    if (bias) {
        if (M <= 1)
            i8gemm_k_ld1_bias_f(A, B_reo, C, A_reorder, &p, bias);
        else if (M <= 2)
            i8gemm_k_ld2_bias_f(A, B_reo, C, A_reorder, &p, bias);
        else if (M <= 4)
            i8gemm_k_ld4_bias_f(A, B_reo, C, A_reorder, &p, bias);
        else
            i8gemm_k_ld_bias_f(A, B_reo, C, A_reorder, &p, bias);
    } else {
        if (M <= 1)
            i8gemm_k_ld1_f(A, B_reo, C, A_reorder, &p);
        else if (M <= 2)
            i8gemm_k_ld2_f(A, B_reo, C, A_reorder, &p);
        else if (M <= 4)
            i8gemm_k_ld4_f(A, B_reo, C, A_reorder, &p);
        else
            i8gemm_k_ld_f(A, B_reo, C, A_reorder, &p);
    }
}

typedef void (*i8_scaled_kernel_t)(const i8_t *, const i8_t *, void *, i8_t *,
                                   const gemm_params_t *, const f32_t *,
                                   const f32_t *, const f32_t *);

static void i8_dispatch_scaled(const i8_t *A, const i8_t *B_reo, void *C,
                               int M, int K_r, int N_r, int ldc,
                               i8_t *A_reorder, const f32_t *row_scale,
                               const f32_t *column_scale, const f32_t *bias,
                               i8_scaled_kernel_t kernel) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    kernel(A, B_reo, C, A_reorder, &p, row_scale, column_scale, bias);
}

void i8gemm_mt_dispatch(const i8_t *A, const i8_t *B_reo,
                        i32_t *C, int M, int K_r, int N_r,
                        int num_threads) {
    const int n_tile = i8_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    // Resolve the requested thread count: num_threads<=0 means "auto" (all
    // cores, or the recommended knee when I8_SVE_AUTO_THREADS=1).
    int req = num_threads > 0 ? num_threads : omp_get_max_threads();
    if (req <= 0) req = 1;
    if (num_threads <= 0) {
        const char *at = getenv("I8_SVE_AUTO_THREADS");
        if (at && atoi(at) != 0)
            req = i8gemm_recommend_threads(M, K_r, N_r, req);
    }
    // Choose the microkernel path on the resolved count, then apply the
    // fork-amortisation clamp. The hybrid no-pack path has a single cheap fork,
    // so it uses a lenient floor (64Ki) that lets small shapes scale to full
    // width on few-core machines; the packed path keeps the 512Ki default that
    // protects high-core-count (80c) boxes from over-subscription collapse.
    const int hybrid = i8_use_hybrid_for_shape(M, K_r, N_r, req);
    num_threads = i8_sve_effective_threads(req, M, K_r, n_tiles,
                                           hybrid ? (64L << 10) : -1);

    // M<=4: the 8-row tile wastes (8-M)/8 of every smmla. Route to a narrow
    // kernel that processes only the needed row-pairs: M<=2 -> 1 pair (4
    // smmla/K8), M=3/4 -> 2 pairs (8 smmla/K8). Measured on V1 vs the 8-row
    // hybrid: M3 +24%, M4 +22% (M>=5 the hybrid's wider B-amortisation wins).
    // Same B_reo layout; int32 path only; N-split for multi-thread.
    if (M <= 4) {
        void (*kern)(const i8_t *, const i8_t *, i32_t *, i8_t *,
                     const gemm_params_t *) =
            (M <= 2) ? i8gemm_k_narrow : i8gemm_k_narrow2;
        if (num_threads <= 1) {
            gemm_params_t p = {M, K_r, N_r, K_r, K_r, N_r};
            kern(A, B_reo, C, NULL, &p);
        } else {
            int nt = n_tiles;
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int t = 0; t < nt; t++) {
                gemm_params_t pp = {M, K_r, n_tile, K_r, K_r, N_r};
                kern(A, B_reo + (size_t)t * K_r * n_tile,
                     C + (size_t)t * n_tile, NULL, &pp);
            }
        }
        return;
    }

    if (hybrid) {
        if (num_threads <= 1) {
            gemm_params_t p = {M, K_r, N_r, K_r, K_r, N_r};
            i8gemm_k_hybrid(A, B_reo, C, NULL, &p);
        } else if (M <= 8) {
            // Too few rows to M-split: split the N tiles across threads.
            int nt = N_r / n_tile;
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int t = 0; t < nt; t++) {
                gemm_params_t pp = {M, K_r, n_tile, K_r, K_r, N_r};
                i8gemm_k_hybrid(A, B_reo + (size_t)t * K_r * n_tile,
                                C + (size_t)t * n_tile, NULL, &pp);
            }
        } else {
            int mblk = (M + 7) / 8;          // total 8-row blocks
            int nt = N_r / n_tile;
            // Pick a balanced, traffic-optimal 2D thread grid pm x pn that
            // divides BOTH the m-block and n-panel counts evenly. gpm/gpn stay
            // 0 when no even grid exists.
            int gpm = 1, gpn = 1;
            i8_pick_grid(mblk, nt, num_threads, M, N_r, &gpm, &gpn);
            // A pure M-split only has `mblk` units of parallelism; once that is
            // below the thread count most threads idle (the high-core-count
            // collapse), so engage the 2D grid even for small shapes there.
            const int starved = (mblk < num_threads);
            // Default distribution heuristic:
            //   * an even 2D grid with pn>1 (mode 4) when one exists -> best
            //     balance + least bandwidth (192/224/256: +1..5%);
            //   * else 2D 8-row x n_tile collapse (mode 2) when mblk does not
            //     divide num_threads -> rebalances the M-split straggler
            //     (200/208/240: +8..23%);
            //   * else pure M-split (mode 0) -> best full-N inner locality.
            // I8_SVE_HYBMT overrides for experimentation
            // (0=M-split 1=N-split 2=2D-collapse 3=balanced-band 4=2D-grid).
            // Only deviate from M-split once the shape is large enough that
            // rebalancing/bandwidth gains outweigh the extra OMP overhead and
            // smaller inner tiles. Below ~128 in either dim, plain M-split is
            // fastest (measured on V1: 64/96/112 cubes regress with grid/2D).
            const int big = (M >= 128 && N_r >= 128);
            const char *mtenv = getenv("I8_SVE_HYBMT");
            int mtmode = mtenv ? atoi(mtenv)
                       : (gpn > 1 && (big || starved)) ? 4
                       : (!big ? 0
                          : (mblk % num_threads != 0 ? 2 : 0));
            if (mtmode == 4 && gpn <= 1) {   // env-forced grid: derive factors
                gpn = (num_threads % 2 == 0 && nt >= 2) ? 2 : 1;
                gpm = num_threads / gpn;
            }
            if (mtmode == 1) {
                // N-split: each thread owns a group of N-panels and sweeps all
                // M rows. The B-panel (K_r x n_tile) stays L1-resident while A
                // streams, instead of every thread re-streaming the full B.
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                for (int t = 0; t < nt; t++) {
                    gemm_params_t pp = {M, K_r, n_tile, K_r, K_r, N_r};
                    i8gemm_k_hybrid(A, B_reo + (size_t)t * K_r * n_tile,
                                    C + (size_t)t * n_tile, NULL, &pp);
                }
            } else if (mtmode == 2) {
                // 2D (M x N) blocking: flatten (n-panel, m-block) with n outer
                // so a static contiguous chunk per thread is N-major -> B-panel
                // reuse across the m-blocks a thread holds. Balances better than
                // pure M- or N-split when M/8 or nt is not a multiple of P.
                long total = (long)nt * mblk;
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                for (long it = 0; it < total; it++) {
                    int t = (int)(it / mblk);
                    int mi = (int)(it % mblk);
                    int m0 = mi * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    gemm_params_t pp = {mb, K_r, n_tile, K_r, K_r, N_r};
                    i8gemm_k_hybrid(A + (size_t)m0 * K_r,
                                    B_reo + (size_t)t * K_r * n_tile,
                                    C + (size_t)m0 * N_r + (size_t)t * n_tile,
                                    NULL, &pp);
                }
            } else if (mtmode == 3) {
                // Balanced contiguous M-band split: partition M into one nearly
                // equal contiguous row-band per thread (rounded to 8-row units),
                // instead of round-robining 8-row blocks. Removes the load
                // imbalance that pure 8-row M-split has when M/8 is not a
                // multiple of P, while keeping full-N inner locality.
                int q = mblk / num_threads;
                int rem = mblk % num_threads;
                #pragma omp parallel num_threads(num_threads)
                {
                    int tid = omp_get_thread_num();
                    int blk0 = tid * q + (tid < rem ? tid : rem);
                    int nblk = q + (tid < rem ? 1 : 0);
                    int m0 = blk0 * 8;
                    int m1 = (blk0 + nblk) * 8;
                    if (m1 > M) m1 = M;
                    if (m0 < m1) {
                        gemm_params_t pp = {m1 - m0, K_r, N_r, K_r, K_r, N_r};
                        i8gemm_k_hybrid(A + (size_t)m0 * K_r, B_reo,
                                        C + (size_t)m0 * N_r, NULL, &pp);
                    }
                }
            } else if (mtmode == 4) {
                // 2D thread grid (pm x pn): each thread owns a contiguous
                // M-band AND a contiguous N-band. A-band is shared by pn
                // threads, B-band by pm threads, so aggregate LLC/DRAM traffic
                // is pn*|A| + pm*|B| vs M-split's |A| + P*|B|. For P=8 a 4x2
                // grid roughly halves it -> helps when bandwidth-bound.
                int pn = gpn, pm = gpm;
                // Spawn exactly pm*pn workers (<= num_threads): with an uneven
                // grid pm*pn may be < num_threads, and any tid >= pm*pn would
                // map past the n-tile grid and write out of bounds.
                #pragma omp parallel num_threads(pm * pn)
                {
                    int tid = omp_get_thread_num();
                    int pr = tid % pm, pc = tid / pm;
                    int bq = mblk / pm, br = mblk % pm;
                    int b0 = pr * bq + (pr < br ? pr : br);
                    int nb = bq + (pr < br ? 1 : 0);
                    int tq = nt / pn, tr = nt % pn;
                    int t0 = pc * tq + (pc < tr ? pc : tr);
                    int ntb = tq + (pc < tr ? 1 : 0);
                    int m0 = b0 * 8, m1 = (b0 + nb) * 8;
                    if (m1 > M) m1 = M;
                    int n0 = t0 * n_tile, ncols = ntb * n_tile;
                    if (m0 < m1 && ncols > 0) {
                        gemm_params_t pp = {m1 - m0, K_r, ncols, K_r, K_r, N_r};
                        i8gemm_k_hybrid(A + (size_t)m0 * K_r,
                                        B_reo + (size_t)t0 * K_r * n_tile,
                                        C + (size_t)m0 * N_r + n0, NULL, &pp);
                    }
                }
            } else {
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                for (int m0 = 0; m0 < M; m0 += 8) {
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    gemm_params_t pp = {mb, K_r, N_r, K_r, K_r, N_r};
                    i8gemm_k_hybrid(A + (size_t)m0 * K_r, B_reo,
                                    C + (size_t)m0 * N_r, NULL, &pp);
                }
            }
        }
        return;
    }

    const int use_n_split =
        i8_sve_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    const int use_a_reorder = i8_use_a_reorder_for_shape(K_r, n_tiles);
    const size_t a_stride = i8_a_reorder_stride(K_r);
    const int m12_allowed = use_a_reorder && i8_use_m12_for_shape(M, K_r, n_tiles);
    const int m12_blocks = m12_allowed ? i8_m12_block_count(M) : 0;
    const int use_m12 = m12_blocks > 0;
    const int m12_rows = m12_blocks * 12;
    const int tail_M = M - m12_rows;
    const size_t a_stride_m12 = i8_a_reorder_stride_m12(K_r);
    i8_t *A_reo_m12_pool = use_m12 ?
        i8_prepare_A_reorder_pool_m12(A, m12_rows, K_r, num_threads,
            i8_scratch(&g_a_poolA, &g_a_capA,
                       (size_t)m12_blocks * a_stride_m12)) : NULL;
    i8_t *A_reo_tail_pool = use_a_reorder && tail_M > 0 ?
        i8_prepare_A_reorder_pool(A + (size_t)m12_rows * K_r,
                                  tail_M, K_r, num_threads,
            i8_scratch(&g_a_poolB, &g_a_capB,
                       (size_t)((tail_M + 7) / 8) * a_stride)) : NULL;
    i8_t *A_reo_pool = use_a_reorder && !use_m12 ?
        i8_prepare_A_reorder_pool(A, M, K_r, num_threads,
            i8_scratch(&g_a_poolA, &g_a_capA,
                       (size_t)((M + 7) / 8) * a_stride)) : NULL;

    /*
     * 2D thread grid over (row-block x n-tile) for the packed path. Pure
     * full-N M-split makes every thread re-stream all of B; a balanced grid
     * (each thread owns an M-band and an N-band) cuts aggregate A/B traffic to
     * pn*|A| + pm*|B| and removes the straggler when the block count does not
     * divide num_threads. Reuses the per-row-block A-reorder prepared above;
     * row-blocks are the m12 12-row blocks followed by the 8-row tail blocks.
     * The grid is uneven (pm*pn <= num_threads, remainder bands spread one
     * extra unit each), so it stays balanced for non-power-of-2 / prime thread
     * counts -- the case that made a pure M-split over ~6 row-blocks collapse
     * at 79/80 threads. Falls back to the existing M/N-split when no useful
     * 2D grid exists (gpn<=1). I8_SVE_PACK2D=0 forces the old path.
     */
    const char *pack2d_env = getenv("I8_SVE_PACK2D");
    int pack2d = !(pack2d_env && atoi(pack2d_env) == 0);
    int nb_rows = use_m12 ? (m12_blocks + (tail_M + 7) / 8) : ((M + 7) / 8);
    int gpm = 0, gpn = 0;
    /*
     * Only deviate from the existing M-split / n-split here when the 2D grid
     * is predicted to win, to avoid regressing shapes the old paths handle
     * well. Two triggers (measured on V1):
     *   (a) load balance -- too few row-blocks to fill all threads with a pure
     *       M-split (nb_rows < num_threads), so some threads idle;
     *   (b) B re-streaming -- in M-split each thread re-reads the full B
     *       (K*N bytes) once per m-block it owns; when that per-thread volume
     *       is large the grid's N-banding cuts it by pn.
     * Leave the tuned n-split path (use_n_split) untouched: overriding it
     * regressed large-N / large-B shapes.
     */
    int per_thread_blocks = (nb_rows + num_threads - 1) / num_threads;
    size_t restream_bytes = (size_t)per_thread_blocks * (size_t)K_r * (size_t)N_r;
    int grid_worth = (nb_rows < num_threads) ||
                     (restream_bytes >= (256u << 10));
    /*
     * Narrow-N guard. The 2D grid's only benefit is cutting B re-streaming by
     * N-banding, which needs enough N-tiles to band. With few n_tiles the picker
     * is forced into a tall pm x small-pn grid that bands N into thin column
     * strips and load-imbalances the row dimension -- measured on Huawei 80c this
     * is a large LOSS for narrow N (2048x4096x64 @t48: grid 4573 vs M-split 9024,
     * +97%; 2048x16384x24 @t64: 3153 vs 5649, +79%). Require n_tiles >= 8 so the
     * grid only runs where N-banding actually pays; narrow N falls back to the
     * pure M-split, which streams B from L2 per row-band and wins. (t79 rescue
     * shapes have large N -> n_tiles >= 64, unaffected.)
     */
    if (pack2d && num_threads > 1 && use_a_reorder && n_tiles >= 8 &&
        !use_n_split && grid_worth)
        i8_pick_grid(nb_rows, n_tiles, num_threads, M, N_r, &gpm, &gpn);

    if (gpn > 1) {
        // Spawn exactly pm*pn workers (<= num_threads) so every tid maps to a
        // valid band; remaining row-blocks / n-tiles are spread one-per-band.
        int gthreads = gpm * gpn;
        #pragma omp parallel num_threads(gthreads)
        {
            int tid = omp_get_thread_num();
            int pr = tid % gpm, pc = tid / gpm;
            int bq = nb_rows / gpm, brem = nb_rows % gpm;
            int r0 = pr * bq + (pr < brem ? pr : brem);
            int r1 = r0 + bq + (pr < brem ? 1 : 0);
            int tq = n_tiles / gpn, trem = n_tiles % gpn;
            int t0 = pc * tq + (pc < trem ? pc : trem);
            int t1 = t0 + tq + (pc < trem ? 1 : 0);
            int n0 = t0 * n_tile;
            int ncols = (t1 - t0) * n_tile;
            const i8_t *B_tile = B_reo + (size_t)t0 * K_r * n_tile;
            for (int r = r0; r < r1; r++) {
                if (use_m12 && r < m12_blocks) {
                    int m0 = r * 12;
                    i8_dispatch_i32_m12(A + (size_t)m0 * K_r, B_tile,
                                        C + (size_t)m0 * N_r + n0,
                                        K_r, ncols, N_r,
                                        A_reo_m12_pool + (size_t)r * a_stride_m12);
                } else {
                    int rt = use_m12 ? (r - m12_blocks) : r;
                    int m0 = (use_m12 ? m12_rows : 0) + rt * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    i8_t *pool = use_m12 ? A_reo_tail_pool : A_reo_pool;
                    i8_t *areo = pool ? pool + (size_t)rt * a_stride : NULL;
                    i8_dispatch_i32(A + (size_t)m0 * K_r, B_tile,
                                    C + (size_t)m0 * N_r + n0,
                                    mb, K_r, ncols, N_r, 1, areo);
                }
            }
        }
    } else if (!use_n_split) {
        if (use_m12) {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int b = 0; b < m12_blocks; b++) {
                int m0 = b * 12;
                i8_dispatch_i32_m12(A + (size_t)m0 * K_r, B_reo,
                                    C + (size_t)m0 * N_r,
                                    K_r, N_r, N_r,
                                    A_reo_m12_pool + (size_t)b * a_stride_m12);
            }

            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int m0 = m12_rows; m0 < M; m0 += 8) {
                int rel_m = m0 - m12_rows;
                int mb = M - m0 < 8 ? M - m0 : 8;
                i8_t *A_reo = A_reo_tail_pool ?
                    A_reo_tail_pool + (size_t)(rel_m / 8) * a_stride : NULL;
                i8_dispatch_i32(A + (size_t)m0 * K_r, B_reo,
                                C + (size_t)m0 * N_r, mb, K_r, N_r, N_r,
                                1, A_reo);
            }
        } else {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int m0 = 0; m0 < M; m0 += 8) {
                int mb = M - m0 < 8 ? M - m0 : 8;
                i8_t *A_reo = A_reo_pool ?
                    A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
                i8_dispatch_i32(A + (size_t)m0 * K_r, B_reo,
                                C + (size_t)m0 * N_r, mb, K_r, N_r, N_r,
                                1, A_reo);
            }
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            const i8_t *B_tile = B_reo + (size_t)t * K_r * n_tile;
            if (use_m12) {
                for (int b = 0; b < m12_blocks; b++) {
                    int m0 = b * 12;
                    i8_dispatch_i32_m12(A + (size_t)m0 * K_r, B_tile,
                                        C + (size_t)m0 * N_r + n0,
                                        K_r, n_tile, N_r,
                                        A_reo_m12_pool + (size_t)b * a_stride_m12);
                }
                for (int m0 = m12_rows; m0 < M; m0 += 8) {
                    int rel_m = m0 - m12_rows;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    i8_t *A_reo = A_reo_tail_pool ?
                        A_reo_tail_pool + (size_t)(rel_m / 8) * a_stride : NULL;
                    i8_dispatch_i32(A + (size_t)m0 * K_r, B_tile,
                                    C + (size_t)m0 * N_r + n0,
                                    mb, K_r, n_tile, N_r, 1, A_reo);
                }
            } else {
                i8_dispatch_i32(A, B_tile, C + n0, M, K_r, n_tile,
                                N_r, 1, A_reo_pool);
            }
        }
    }

    /* A-reorder pools live in thread-local cached scratch; do not free. */
}

void i8gemm_mt_dispatch_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, int M, int K_r, int N_r,
                          int num_threads) {
    const int n_tile = i8_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = i8_sve_effective_threads(num_threads, M, K_r, n_tiles, -1);
    const int use_n_split =
        i8_sve_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    const int use_a_reorder = i8_use_a_reorder_for_shape(K_r, n_tiles);
    const size_t a_stride = i8_a_reorder_stride(K_r);
    i8_t *A_reo_pool = use_a_reorder ?
        i8_prepare_A_reorder_pool(A, M, K_r, num_threads,
            i8_scratch(&g_a_poolA, &g_a_capA,
                       (size_t)((M + 7) / 8) * a_stride)) : NULL;

    if (!use_n_split) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            i8_t *A_reo = A_reo_pool ?
                A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
            i8_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                            C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, NULL,
                            A_reo);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            i8_dispatch_f32(A, B_reo + (size_t)t * K_r * n_tile,
                            C + n0, M, K_r, n_tile, N_r, NULL, A_reo_pool);
        }
    }

    /* cached scratch; do not free. */
}

typedef struct {
    const i8_t *B_reo;
    void *C;
    const f32_t *column_scale;
    const f32_t *bias;
    int N_r;
} i8_scaled_work_t;

static void i8gemm_mt_dispatch_scaled_pool(
        const i8_t *A, int M, int K_r, int num_threads,
        const f32_t *row_scale, i8_scaled_work_t *work, int work_count,
        int output_shift, i8_scaled_kernel_t kernel_m8,
        i8_scaled_kernel_t kernel_m12) {
    const int n_tile = i8_sve_n_tile();
    int total_n_tiles = 0;
    int max_n_tiles = 0;
    int group_count[2] = {0, 0};
    const int tiles_per_window =
        (1 << 20) / (K_r * n_tile) > 0 ? (1 << 20) / (K_r * n_tile) : 1;
    for (int index = 0; index < work_count; index++) {
        int tiles = work[index].N_r / n_tile;
        total_n_tiles += tiles;
        if (tiles > max_n_tiles)
            max_n_tiles = tiles;
        group_count[index] = (tiles + tiles_per_window - 1) / tiles_per_window;
    }
    num_threads = i8_sve_effective_threads(
        num_threads, M, K_r, total_n_tiles > 0 ? total_n_tiles : 1, -1);

    const int use_m12 = i8_use_m12_for_shape(M, K_r, max_n_tiles);
    const int m12_blocks = use_m12 ? i8_m12_block_count(M) : 0;
    const int m12_rows = m12_blocks * 12;
    const int tail_M = M - m12_rows;
    const int tail_blocks = (tail_M + 7) / 8;
    const int row_blocks = m12_blocks + tail_blocks;
    const size_t stride_m12 = i8_a_reorder_stride_m12(K_r);
    const size_t stride_m8 = i8_a_reorder_stride(K_r);
    i8_t *packed_m12 = m12_blocks > 0 ?
        i8_scratch(&g_a_poolA, &g_a_capA,
                   (size_t)m12_blocks * stride_m12) : NULL;
    i8_t *packed_tail = tail_blocks > 0 ?
        i8_scratch(&g_a_poolB, &g_a_capB,
                   (size_t)tail_blocks * stride_m8) : NULL;
    int total_groups = group_count[0] + (work_count > 1 ? group_count[1] : 0);
    int total_tasks = total_groups * row_blocks;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static)
        for (int block = 0; block < m12_blocks; block++) {
            i8_pack_A_sve_m12_block(A + (size_t)block * 12u * K_r,
                                    packed_m12 + (size_t)block * stride_m12,
                                    K_r);
        }
        #pragma omp for schedule(static)
        for (int block = 0; block < tail_blocks; block++) {
            int m0 = m12_rows + block * 8;
            int mb = M - m0 < 8 ? M - m0 : 8;
            i8_pack_A_sve_block(A + (size_t)m0 * K_r,
                                packed_tail + (size_t)block * stride_m8,
                                mb, K_r);
        }
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < total_tasks; task++) {
            int global_group = task / row_blocks;
            int row_block = task % row_blocks;
            int work_index = global_group >= group_count[0] ? 1 : 0;
            int local_group = global_group - (work_index ? group_count[0] : 0);
            i8_scaled_work_t *item = &work[work_index];
            int n_tiles = item->N_r / n_tile;
            int tile_begin = local_group * tiles_per_window;
            int tile_end = tile_begin + tiles_per_window;
            if (tile_end > n_tiles)
                tile_end = n_tiles;
            int n0 = tile_begin * n_tile;
            int ncols = (tile_end - tile_begin) * n_tile;
            const i8_t *B_window =
                item->B_reo + (size_t)tile_begin * K_r * n_tile;
            uint8_t *C_bytes = (uint8_t *)item->C;
            if (row_block < m12_blocks) {
                int m0 = row_block * 12;
                size_t c_offset =
                    ((size_t)m0 * item->N_r + (size_t)n0) << output_shift;
                i8_dispatch_scaled(
                    A + (size_t)m0 * K_r, B_window, C_bytes + c_offset,
                    12, K_r, ncols, item->N_r,
                    packed_m12 + (size_t)row_block * stride_m12,
                    row_scale + m0, item->column_scale + n0,
                    item->bias ? item->bias + n0 : NULL, kernel_m12);
            } else {
                int tail_block = row_block - m12_blocks;
                int m0 = m12_rows + tail_block * 8;
                int mb = M - m0 < 8 ? M - m0 : 8;
                size_t c_offset =
                    ((size_t)m0 * item->N_r + (size_t)n0) << output_shift;
                i8_dispatch_scaled(
                    A + (size_t)m0 * K_r, B_window, C_bytes + c_offset,
                    mb, K_r, ncols, item->N_r,
                    packed_tail + (size_t)tail_block * stride_m8,
                    row_scale + m0, item->column_scale + n0,
                    item->bias ? item->bias + n0 : NULL, kernel_m8);
            }
        }
    }
}

void i8gemm_mt_dispatch_scaled_f(const i8_t *A, const i8_t *B_reo,
                                  f32_t *C, int M, int K_r, int N_r,
                                  int num_threads, const f32_t *row_scale,
                                  const f32_t *column_scale,
                                  const f32_t *bias) {
    i8_scaled_work_t work = {B_reo, C, column_scale, bias, N_r};
    i8gemm_mt_dispatch_scaled_pool(A, M, K_r, num_threads, row_scale,
                                   &work, 1, 2, i8gemm_k_scaled_f,
                                   i8gemm_k_scaled_m12_f);
}

void i8gemm_mt_dispatch_scaled_b(const i8_t *A, const i8_t *B_reo,
                                  uint16_t *C, int M, int K_r, int N_r,
                                  int num_threads, const f32_t *row_scale,
                                  const f32_t *column_scale,
                                  const f32_t *bias) {
    i8_scaled_work_t work = {B_reo, C, column_scale, bias, N_r};
    i8gemm_mt_dispatch_scaled_pool(A, M, K_r, num_threads, row_scale,
                                   &work, 1, 1, i8gemm_k_scaled_b,
                                   i8gemm_k_scaled_m12_b);
}

void i8gemm_mt_dispatch_scaled_pair_b(
        const i8_t *A, const i8_t *first_B_reo, uint16_t *first_C,
        int first_N_r, const f32_t *first_column_scale,
        const f32_t *first_bias, const i8_t *second_B_reo,
        uint16_t *second_C, int second_N_r,
        const f32_t *second_column_scale, const f32_t *second_bias,
        int M, int K_r, int num_threads, const f32_t *row_scale) {
    i8_scaled_work_t work[2] = {
        {first_B_reo, first_C, first_column_scale, first_bias, first_N_r},
        {second_B_reo, second_C, second_column_scale, second_bias, second_N_r},
    };
    i8gemm_mt_dispatch_scaled_pool(A, M, K_r, num_threads, row_scale,
                                   work, 2, 1, i8gemm_k_scaled_b,
                                   i8gemm_k_scaled_m12_b);
}

void i8gemm_mt_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                               f32_t *C, int M, int K_r, int N_r,
                               int num_threads, const f32_t *bias) {
    const int n_tile = i8_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = i8_sve_effective_threads(num_threads, M, K_r, n_tiles, -1);
    const int use_n_split =
        i8_sve_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    const int use_a_reorder = i8_use_a_reorder_for_shape(K_r, n_tiles);
    const size_t a_stride = i8_a_reorder_stride(K_r);
    i8_t *A_reo_pool = use_a_reorder ?
        i8_prepare_A_reorder_pool(A, M, K_r, num_threads,
            i8_scratch(&g_a_poolA, &g_a_capA,
                       (size_t)((M + 7) / 8) * a_stride)) : NULL;

    if (!use_n_split) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            i8_t *A_reo = A_reo_pool ?
                A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
            i8_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                            C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, bias,
                            A_reo);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            i8_dispatch_f32(A, B_reo + (size_t)t * K_r * n_tile,
                            C + n0, M, K_r, n_tile, N_r, bias + n0,
                            A_reo_pool);
        }
    }

    /* cached scratch; do not free. */
}

static int i8_prepare(const i8_t *A_orig, const i8_t *B_orig,
                      i8_t **A_pad, i8_t **B_reo,
                      int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 16 ? 16 : K, 16);
    *N_r = round_up_int(N < 8 ? 8 : N, i8_sve_n_tile());
    *A_pad = NULL;
    *B_reo = NULL;

    i8_t *B_pad = (i8_t *)calloc((size_t)*K_r * *N_r, sizeof(i8_t));
    if (!B_pad)
        return 0;
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B_orig + (size_t)i * N, (size_t)N);

    *B_reo = (i8_t *)aligned_alloc_64((size_t)*K_r * *N_r);
    if (!*B_reo) {
        free(B_pad);
        return 0;
    }
    i8_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);

    if (*K_r == K) {
        *A_pad = (i8_t *)A_orig;
    } else {
        *A_pad = (i8_t *)calloc((size_t)M * *K_r, sizeof(i8_t));
        if (!*A_pad) {
            free(*B_reo);
            return 0;
        }
        for (int i = 0; i < M; i++)
            memcpy(*A_pad + (size_t)i * *K_r, A_orig + (size_t)i * K, (size_t)K);
    }
    return 1;
}

void i8gemm_mt(const i8_t *A_orig, const i8_t *B_orig,
               i32_t *C, int M, int K, int N, int num_threads) {
    i8_t *A_use, *B_reo;
    int K_r, N_r;
    if (!i8_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    i32_t *C_pad = (i32_t *)calloc((size_t)M * N_r, sizeof(i32_t));
    if (C_pad) {
        i8gemm_mt_dispatch(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(i32_t));
        free(C_pad);
    }
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}

void i8gemm_mt_f(const i8_t *A_orig, const i8_t *B_orig,
                 f32_t *C, int M, int K, int N, int num_threads) {
    i8_t *A_use, *B_reo;
    int K_r, N_r;
    if (!i8_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
    if (C_pad) {
        i8gemm_mt_dispatch_f(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(f32_t));
        free(C_pad);
    }
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}

void i8gemm_mt_bias_f(const i8_t *A_orig, const i8_t *B_orig,
                      f32_t *C, int M, int K, int N,
                      int num_threads, const f32_t *bias) {
    i8_t *A_use, *B_reo;
    int K_r, N_r;
    if (!i8_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    f32_t *bias_pad = (f32_t *)calloc((size_t)N_r, sizeof(f32_t));
    f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
    if (bias_pad && C_pad) {
        memcpy(bias_pad, bias, (size_t)N * sizeof(f32_t));
        i8gemm_mt_dispatch_bias_f(A_use, B_reo, C_pad, M, K_r, N_r,
                                  num_threads, bias_pad);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(f32_t));
    }
    free(bias_pad);
    free(C_pad);
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}
