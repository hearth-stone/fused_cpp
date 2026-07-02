// bf16gemm_silu_packc.S  (fused_cpp-owned; do NOT edit i8gemm)
// Self-contained copy of the ldp_silu kernel machinery from
// refs/i8gemm/lib/bf16gemm_k.S, with a packed-C (reorder-m8) store so the
// w13 fused SiLU epilogue writes intermediate directly in the layout w2
// consumes as pre-packed A (Part 2 of packA fusion). Only NEW global
// symbols (bf16gemm_k_ldp_silu_poly*_packc) are emitted; all shared macros
// are file-local so there is no clash with bf16gemm_k.S in the same link.
//
// packed-C addressing: reuse GEMM_BODY unchanged. M-block stride LDC_B<<3
// (= ldc*16 = 8*F_pad*2 bytes) already matches the reorder-m8 M-block; set
// c_adv=64 so C_N_BASE steps one reorder K-block (32 uint16) per N-tile.
// The store writes the 8 rows of the current K-block as 64 contiguous bytes
// (row r at +r*8), i.e. offset(r)=r*4 uint16 within the block.

// bf16gemm_k.S
// void bf16gemm_k_ld(const uint16_t *A,            // x0  原始 A, M×K 行主序 bf16
//                    const uint16_t *B_reordered,  // x1  已 pack 的 B
//                    float          *C,            // x2  行主序输出 float32
//                    uint16_t       *A_reorder,    // x3  A 重排输出 buffer
//                    const gemm_params_t *params); // x4  {m,k,n,lda,ldb,ldc} 结构体指针
//
// 循环顺序: M-N-K (最外层M, 中间N, 最内层K)
// A 重排策略:
//   每个 M block 的第一个 N iteration 从原始 A 跨 stride (lda) 读取，
//   写入 A_reorder; 后续 N iteration 直接从 A_reorder 连续读取。
// B: 预先 pack 好（外部调用者负责），N-block stride = ldb * 16 bytes。
//
// 内核每次处理 8×8 输出块，v9~v24 的 bfmmla 格式：
//   bfmmla computes: 2×4 bf16 (vn.8h) × 4×2 bf16 (vm.8h) → 2×2 f32 accumulate (vd.4s)
//   v9  = C[0:2][0:2], v10 = C[0:2][2:4], v11 = C[0:2][4:6], v12 = C[0:2][6:8]
//   v13-v16: r2/r3 同上, v17-v20: r4/r5, v21-v24: r6/r7

	.text
	.align 4
#ifdef __APPLE__
#else
#endif

#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA_B       x7    // A row stride in bytes (= lda * 2, computed in init)
#define LDC_B       x9    // C row stride in bytes = ldc * 4
#define LDB         x8    // K_big of packed B (loaded from stack)

#define N_COUNT     x30
#define C_N_BASE    x27
#define M_COUNT     x10
#define B_ADDR_1    x19
#define B_N_BASE    x22
#define K_COUNT     x23
#define FIRST_N     x24   // flag: 1 = 当前 M block 的首个 N iteration
#define C_ROW_BASE  x25
#define C_M_BASE    x26
#define A_REORDER_M x28   // 当前 M block 在 A_reorder 中的基地址
#define A_ADDR      x29   // A 加载指针（指向 orig A 或 reorder buffer）
#define A_ORIG_M    x20 
// ── M_LOOP 宏：m=8 用真循环，m=1/2/4 用空宏 ──────────────────────
	.macro M_LOOP_BEGIN_1
	.endm
	.macro M_LOOP_BEGIN_2
	.endm
	.macro M_LOOP_BEGIN_4
	.endm
	.macro M_LOOP_BEGIN_8
	asr  M_COUNT, M, #3
	cmp  M_COUNT, #0
	b.eq 301f
400:
	.endm

	.macro M_LOOP_END_1
	.endm
	.macro M_LOOP_END_2
	.endm
	.macro M_LOOP_END_4
	.endm
	.macro M_LOOP_END_8
	add  A_ORIG_M, A_ORIG_M, LDA_B, lsl 3
	add  C_ROW_BASE, C_ROW_BASE, LDC_B, lsl 3
	subs M_COUNT, M_COUNT, #1
	b.gt 400b
301:
	.endm

  // 当前 M block 在原始 A 中的基地址


// ── INIT / RESTORE ────────────────────────────────────────────
	.macro INIT
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp x23, x24, [sp, #-16]!
	stp x25, x26, [sp, #-16]!
	stp x27, x28, [sp, #-16]!
	stp x29, x30, [sp, #-16]!
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	.endm

	.macro RESTORE
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ldp x29, x30, [sp], #16
	ldp x27, x28, [sp], #16
	ldp x25, x26, [sp], #16
	ldp x23, x24, [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16
	.endm

// ── Common data-layout helpers ────────────────────────────────────
	.macro LOAD_BF16_ROW_F32 addr, vlo, vhi
	ldr  q0, [\addr]
	shll  \vlo\().4s, v0.4h, #16
	shll2 \vhi\().4s, v0.8h, #16
	.endm

	.macro ZIP12_D dst1, dst2, src1, src2
	zip1 \dst1\().2d, \src1\().2d, \src2\().2d
	zip2 \dst2\().2d, \src1\().2d, \src2\().2d
	.endm

	// Load the PC-relative address of a local const label into \reg
	// (Mach-O @PAGE/@PAGEOFF vs ELF adrp/:lo12:).
	.macro ADR_CONST reg, label
#ifdef __APPLE__
	adrp \reg, \label@PAGE
	add  \reg, \reg, \label@PAGEOFF
#else
	adrp \reg, \label
	add  \reg, \reg, :lo12:\label
#endif
	.endm

	.macro LOAD_ZERO_MAIN_8
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	movi v13.4s, #0
	movi v14.4s, #0
	movi v15.4s, #0
	movi v16.4s, #0
	movi v17.4s, #0
	movi v18.4s, #0
	movi v19.4s, #0
	movi v20.4s, #0
	movi v21.4s, #0
	movi v22.4s, #0
	movi v23.4s, #0
	movi v24.4s, #0
	.endm

// ── LOAD_C_ROWMAJOR_B：从 bf16 C 加载 8×8 → blocked v9~v24 ──
	.macro LOAD_C_ROWMAJOR_B
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	add  x15, x11, LDC_B, lsl #2
	add  x16, x12, LDC_B, lsl #2
	add  x17, x15, LDC_B, lsl #1
	LOAD_BF16_ROW_F32 C_M_BASE, v2, v3
	LOAD_BF16_ROW_F32 x11, v4, v5
	ZIP12_D v9,  v10, v2, v4
	ZIP12_D v11, v12, v3, v5
	LOAD_BF16_ROW_F32 x12, v2, v3
	LOAD_BF16_ROW_F32 x13, v4, v5
	ZIP12_D v13, v14, v2, v4
	ZIP12_D v15, v16, v3, v5
	LOAD_BF16_ROW_F32 x14, v2, v3
	LOAD_BF16_ROW_F32 x15, v4, v5
	ZIP12_D v17, v18, v2, v4
	ZIP12_D v19, v20, v3, v5
	LOAD_BF16_ROW_F32 x16, v2, v3
	LOAD_BF16_ROW_F32 x17, v4, v5
	ZIP12_D v21, v22, v2, v4
	ZIP12_D v23, v24, v3, v5
	.endm

// ── LOAD_C_ROWMAJOR：从行主序 C 加载 8×8 float32 → blocked v9~v24 ──
// C row stride = LDC_B (= ldc * 4) bytes
	.macro LOAD_C_ROWMAJOR
	// addr: all from C_M_BASE directly, max depth 3
	add  x11, C_M_BASE, LDC_B             // 1*LDC
	add  x12, C_M_BASE, LDC_B, lsl #1     // 2*LDC
	add  x14, C_M_BASE, LDC_B, lsl #2     // 4*LDC
	sub  x13, x14, LDC_B                  // 3*LDC = 4-1
	add  x15, x11, LDC_B, lsl #2          // 5*LDC = 1+4
	add  x16, x12, LDC_B, lsl #2          // 6*LDC = 2+4
	add  x17, x15, LDC_B, lsl #1          // 7*LDC = 5+2

	ldp  q0, q1, [C_M_BASE]
	ldp  q2, q3, [x11]
	ldp  q4, q5, [x12]
	ldp  q6, q7, [x13]
	ldp  q8, q25, [x14]
	ldp  q26, q27, [x15]
	ldp  q28, q29, [x16]
	ldp  q30, q31, [x17]

	// zip pairs to form 2×2 blocked layout
	ZIP12_D v9,  v10, v0, v2
	ZIP12_D v11, v12, v1, v3

	ZIP12_D v13, v14, v4, v6
	ZIP12_D v15, v16, v5, v7

	ZIP12_D v17, v18, v8, v26
	ZIP12_D v19, v20, v25, v27

	ZIP12_D v21, v22, v28, v30
	ZIP12_D v23, v24, v29, v31
	.endm

// ── STORE_C_ROWMAJOR：将 blocked v9~v24 按行主序写回 C ────────
	.macro STORE_C_ROWMAJOR
	// addr: all from C_M_BASE directly, max depth 3
	add  x11, C_M_BASE, LDC_B             // 1*LDC
	add  x12, C_M_BASE, LDC_B, lsl #1     // 2*LDC
	add  x14, C_M_BASE, LDC_B, lsl #2     // 4*LDC
	sub  x13, x14, LDC_B                  // 3*LDC = 4-1
	add  x15, x11, LDC_B, lsl #2          // 5*LDC = 1+4
	add  x16, x12, LDC_B, lsl #2          // 6*LDC = 2+4
	add  x17, x15, LDC_B, lsl #1          // 7*LDC = 5+2

	ZIP12_D v0, v2, v9,  v10
	ZIP12_D v1, v3, v11, v12

	ZIP12_D v4, v6, v13, v14
	ZIP12_D v5, v7, v15, v16

	ZIP12_D v8,  v26, v17, v18
	ZIP12_D v25, v27, v19, v20

	ZIP12_D v28, v30, v21, v22
	ZIP12_D v29, v31, v23, v24

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	stp  q8, q25, [x14]
	stp  q26, q27, [x15]
	stp  q28, q29, [x16]
	stp  q30, q31, [x17]
	.endm

// ── STORE_C_ROWMAJOR_B：f32→bf16 转换后存储，每行 16 bytes ──────
	.macro STORE_C_ROWMAJOR_B
	ZIP12_D v0, v2, v9,  v10
	ZIP12_D v1, v3, v11, v12
	bfcvtn  v0.4h, v0.4s
	bfcvtn2 v0.8h, v1.4s
	bfcvtn  v1.4h, v2.4s
	bfcvtn2 v1.8h, v3.4s

	ZIP12_D v4, v6, v13, v14
	ZIP12_D v5, v7, v15, v16
	bfcvtn  v2.4h, v4.4s
	bfcvtn2 v2.8h, v5.4s
	bfcvtn  v3.4h, v6.4s
	bfcvtn2 v3.8h, v7.4s

	ZIP12_D v8,  v26, v17, v18
	ZIP12_D v25, v27, v19, v20
	bfcvtn  v4.4h, v8.4s
	bfcvtn2 v4.8h, v25.4s
	bfcvtn  v5.4h, v26.4s
	bfcvtn2 v5.8h, v27.4s

	ZIP12_D v28, v30, v21, v22
	ZIP12_D v29, v31, v23, v24
	bfcvtn  v6.4h, v28.4s
	bfcvtn2 v6.8h, v29.4s
	bfcvtn  v7.4h, v30.4s
	bfcvtn2 v7.8h, v31.4s

	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	add  x15, x11, LDC_B, lsl #2
	add  x16, x12, LDC_B, lsl #2
	add  x17, x15, LDC_B, lsl #1

	str  q0, [C_M_BASE]
	str  q1, [x11]
	str  q2, [x12]
	str  q3, [x13]
	str  q4, [x14]
	str  q5, [x15]
	str  q6, [x16]
	str  q7, [x17]
	.endm

// ── STORE_C_SILU_LINEAR_8：融合门控写回（Task 2，无 silu）─────────────
// 交错权重布局下每个 8 列 tile = 4 gate(cols0-3) + 4 up(cols4-7)。
// 复用 STORE_C_ROWMAJOR 的 ZIP 反交错：每行得到 gate[0:4] 与 up[0:4]，
// 逐 lane 相乘 = gate*up（本任务先不做 silu，仅验证布局/寻址/bf16 写回），
// 转 bf16 后每行写 4 个特征 = 8 bytes（str d）。行 stride = LDC_B (= F_pad*2)。
	.macro STORE_C_SILU_LINEAR_8
	add  x11, C_M_BASE, LDC_B             // row1
	add  x12, C_M_BASE, LDC_B, lsl #1     // row2
	add  x14, C_M_BASE, LDC_B, lsl #2     // row4
	sub  x13, x14, LDC_B                  // row3
	add  x15, x11, LDC_B, lsl #2          // row5
	add  x16, x12, LDC_B, lsl #2          // row6
	add  x17, x15, LDC_B, lsl #1          // row7

	// rows 0-1: gate=v9,v10  up=v11,v12
	ZIP12_D v0, v2, v9,  v10              // v0=row0 gate[0:4], v2=row1 gate[0:4]
	ZIP12_D v1, v3, v11, v12              // v1=row0 up[0:4],   v3=row1 up[0:4]
	fmul v0.4s, v0.4s, v1.4s
	fmul v2.4s, v2.4s, v3.4s
	bfcvtn v0.4h, v0.4s
	bfcvtn v2.4h, v2.4s
	str  d0, [C_M_BASE]
	str  d2, [x11]

	// rows 2-3: gate=v13,v14  up=v15,v16
	ZIP12_D v4, v6, v13, v14
	ZIP12_D v5, v7, v15, v16
	fmul v4.4s, v4.4s, v5.4s
	fmul v6.4s, v6.4s, v7.4s
	bfcvtn v4.4h, v4.4s
	bfcvtn v6.4h, v6.4s
	str  d4, [x12]
	str  d6, [x13]

	// rows 4-5: gate=v17,v18  up=v19,v20
	ZIP12_D v8,  v26, v17, v18
	ZIP12_D v25, v27, v19, v20
	fmul v8.4s,  v8.4s,  v25.4s
	fmul v26.4s, v26.4s, v27.4s
	bfcvtn v8.4h,  v8.4s
	bfcvtn v26.4h, v26.4s
	str  d8,  [x14]
	str  d26, [x15]

	// rows 6-7: gate=v21,v22  up=v23,v24
	ZIP12_D v28, v30, v21, v22
	ZIP12_D v29, v31, v23, v24
	fmul v28.4s, v28.4s, v29.4s
	fmul v30.4s, v30.4s, v31.4s
	bfcvtn v28.4h, v28.4s
	bfcvtn v30.4h, v30.4s
	str  d28, [x16]
	str  d30, [x17]
	.endm

// ── SILU_ROW：单行 (4 特征) 融合 silu-and-mul，exp 多项式阶数 \deg∈{4,5,6}──
// out = g*u / (1 + exp(-g)) = silu(g)*u。port 自 vexpq_f32_poly_impl<deg>。
// 持久常量寄存器（由 STORE_C_SILU_POLY_8 预载）:
//   v25=invln2 v26=ln2 v27=c3 v28=c4 v29=c5 v30=c6 v7=clamp_hi v8=clamp_lo
// c2(0.5)/c1(1)/c0(1)/one 用 fmov 立即数内联（省寄存器，让 c6 放得下）。
// 参数为寄存器编号：g=gate(输入并接收结果), u=up, a/b/c/d=scratch, deg=阶数。
// Horner 头部结束后 poly 恒定落在 v\b；结果 bf16 落在 v\g 低 64 位。
	.macro SILU_ROW g, u, a, b, c, d, deg
	fneg   v\a\().4s, v\g\().4s              // x = -g
	fmin   v\a\().4s, v\a\().4s, v7.4s       // clamp hi
	fmax   v\a\().4s, v\a\().4s, v8.4s       // clamp lo
	fmul   v\b\().4s, v\a\().4s, v25.4s      // x * invln2
	frintn v\b\().4s, v\b\().4s              // fn = round-to-even
	fcvtzs v\c\().4s, v\b\().4s              // n = (int)fn
	fmls   v\a\().4s, v\b\().4s, v26.4s      // a = r = x - fn*ln2
	.if \deg == 4
	mov    v\b\().16b, v28.16b               // poly = c4
	.elseif \deg == 5
	mov    v\d\().16b, v29.16b               // c5
	mov    v\b\().16b, v28.16b               // c4
	fmla   v\b\().4s, v\d\().4s, v\a\().4s   // poly = c4 + c5*r
	.else
	mov    v\b\().16b, v30.16b               // c6
	mov    v\d\().16b, v29.16b               // c5
	fmla   v\d\().4s, v\b\().4s, v\a\().4s   // c5 + c6*r
	mov    v\b\().16b, v28.16b               // c4
	fmla   v\b\().4s, v\d\().4s, v\a\().4s   // poly = c4 + poly*r
	.endif
	// common tail: c3, c2, c1, c0 (poly in v\b on entry and exit)
	mov    v\d\().16b, v27.16b               // c3
	fmla   v\d\().4s, v\b\().4s, v\a\().4s
	fmov   v\b\().4s, #0.5                    // c2
	fmla   v\b\().4s, v\d\().4s, v\a\().4s
	fmov   v\d\().4s, #1.0                    // c1
	fmla   v\d\().4s, v\b\().4s, v\a\().4s
	fmov   v\b\().4s, #1.0                    // c0
	fmla   v\b\().4s, v\d\().4s, v\a\().4s    // poly (in v\b)
	movi   v\d\().4s, #0x7f                    // 127
	add    v\c\().4s, v\c\().4s, v\d\().4s    // n + 127
	shl    v\c\().4s, v\c\().4s, #23          // pow2n bits = (n+127)<<23
	fmul   v\b\().4s, v\b\().4s, v\c\().4s    // e = poly * 2^n = exp(-g)
	fmov   v\d\().4s, #1.0
	fadd   v\b\().4s, v\b\().4s, v\d\().4s    // denom = 1 + e
	fmul   v\g\().4s, v\g\().4s, v\u\().4s    // num = g * u
	fdiv   v\g\().4s, v\g\().4s, v\b\().4s    // out = num / denom
	bfcvtn v\g\().4h, v\g\().4s               // -> bf16 (low 64b)
	.endm

// ── SILU_CONSTS_LOAD：把 silu/exp 常量广播载入持久寄存器（x11 作基址）──
	.macro SILU_CONSTS_LOAD
	ADR_CONST x11, Csilu_poly_consts
	ld1r {v25.4s}, [x11], #4                 // invln2
	ld1r {v26.4s}, [x11], #4                 // ln2
	ld1r {v27.4s}, [x11], #4                 // c3
	ld1r {v28.4s}, [x11], #4                 // c4
	ld1r {v29.4s}, [x11], #4                 // c5
	ld1r {v30.4s}, [x11], #4                 // c6
	ld1r {v7.4s},  [x11], #4                 // clamp_hi 87
	ld1r {v8.4s},  [x11], #4                 // clamp_lo -87
	.endm

// ── STORE_C_SILU_POLY_8：融合 silu-and-mul 写回，阶数 \deg 参数化 ──────
// 复用 STORE_C_ROWMAJOR 的 ZIP 反交错，逐 pair 处理 8 行，每行写 4 特征。
// scratch: ZIP 输出 v0-v3；SILU_ROW temps v4,v5,v6,v9（累加器随 pair 释放）。
	.macro STORE_C_SILU_POLY_8 deg
	SILU_CONSTS_LOAD

	add  x11, C_M_BASE, LDC_B                // row1
	add  x12, C_M_BASE, LDC_B, lsl #1        // row2
	add  x14, C_M_BASE, LDC_B, lsl #2        // row4
	sub  x13, x14, LDC_B                     // row3
	add  x15, x11, LDC_B, lsl #2             // row5
	add  x16, x12, LDC_B, lsl #2             // row6
	add  x17, x15, LDC_B, lsl #1             // row7

	// pair 0: rows 0,1
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [C_M_BASE]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x11]

	// pair 1: rows 2,3
	ZIP12_D v0, v1, v13, v14
	ZIP12_D v2, v3, v15, v16
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [x12]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x13]

	// pair 2: rows 4,5
	ZIP12_D v0, v1, v17, v18
	ZIP12_D v2, v3, v19, v20
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [x14]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x15]

	// pair 3: rows 6,7
	ZIP12_D v0, v1, v21, v22
	ZIP12_D v2, v3, v23, v24
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [x16]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x17]
	.endm

	// Thin per-degree wrappers so GEMM_BODY can name a store macro.
	.macro STORE_C_SILU_POLY4_8
	STORE_C_SILU_POLY_8 4
	.endm
	.macro STORE_C_SILU_POLY5_8
	STORE_C_SILU_POLY_8 5
	.endm
	.macro STORE_C_SILU_POLY6_8
	STORE_C_SILU_POLY_8 6
	.endm

// ── STORE_C_SILU_POLY_PACKC_8: reorder-m8 packed-C store ─────────────
// Same silu-and-mul math as STORE_C_SILU_POLY_8, but the 8 rows x 4 features
// of the current feature K-block are written as 64 contiguous bytes at
// C_M_BASE (row r at +r*8). Pairs (r0,r1) -> q0=[row0|row1] via ins.
	.macro STORE_C_SILU_POLY_PACKC_8 deg
	SILU_CONSTS_LOAD
	// pair 0: rows 0,1  (accumulators v9,v10,v11,v12)
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE]
	// pair 1: rows 2,3
	ZIP12_D v0, v1, v13, v14
	ZIP12_D v2, v3, v15, v16
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE, #16]
	// pair 2: rows 4,5
	ZIP12_D v0, v1, v17, v18
	ZIP12_D v2, v3, v19, v20
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE, #32]
	// pair 3: rows 6,7
	ZIP12_D v0, v1, v21, v22
	ZIP12_D v2, v3, v23, v24
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE, #48]
	.endm

	.macro STORE_C_SILU_POLY4_PACKC_8
	STORE_C_SILU_POLY_PACKC_8 4
	.endm
	.macro STORE_C_SILU_POLY5_PACKC_8
	STORE_C_SILU_POLY_PACKC_8 5
	.endm
	.macro STORE_C_SILU_POLY6_PACKC_8
	STORE_C_SILU_POLY_PACKC_8 6
	.endm

// ── M-tail fused stores (rows < 8). Mirror STORE_C_{1,2,4}: m=1 uses v9-v12
//    (row1 is a dummy slot, not written); m=2 rows0-1; m=4 rows0-3 (v9-v16). ──
	.macro STORE_C_SILU_POLY_M1 deg
	SILU_CONSTS_LOAD
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [C_M_BASE]
	.endm

	.macro STORE_C_SILU_POLY_M2 deg
	SILU_CONSTS_LOAD
	add  x11, C_M_BASE, LDC_B
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [C_M_BASE]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x11]
	.endm

	.macro STORE_C_SILU_POLY_M4 deg
	SILU_CONSTS_LOAD
	add  x11, C_M_BASE, LDC_B                // row1
	add  x12, C_M_BASE, LDC_B, lsl #1        // row2
	add  x14, C_M_BASE, LDC_B, lsl #2        // 4*LDC
	sub  x13, x14, LDC_B                      // row3 = 4-1
	// pair 0: rows 0,1
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [C_M_BASE]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x11]
	// pair 1: rows 2,3
	ZIP12_D v0, v1, v13, v14
	ZIP12_D v2, v3, v15, v16
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [x12]
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	str  d1, [x13]
	.endm

	.macro STORE_C_SILU_POLY4_M1
	STORE_C_SILU_POLY_M1 4
	.endm
	.macro STORE_C_SILU_POLY5_M1
	STORE_C_SILU_POLY_M1 5
	.endm
	.macro STORE_C_SILU_POLY6_M1
	STORE_C_SILU_POLY_M1 6
	.endm
	.macro STORE_C_SILU_POLY4_M2
	STORE_C_SILU_POLY_M2 4
	.endm
	.macro STORE_C_SILU_POLY5_M2
	STORE_C_SILU_POLY_M2 5
	.endm
	.macro STORE_C_SILU_POLY6_M2
	STORE_C_SILU_POLY_M2 6
	.endm
	.macro STORE_C_SILU_POLY4_M4
	STORE_C_SILU_POLY_M4 4
	.endm
	.macro STORE_C_SILU_POLY5_M4
	STORE_C_SILU_POLY_M4 5
	.endm
	.macro STORE_C_SILU_POLY6_M4
	STORE_C_SILU_POLY_M4 6
	.endm

// ==================== CACHED PATH MACROS ====================
// 从 A_reorder 连续读取 A（后续 N iteration 使用）
//
// A 加载: 4 regs = 8 rows × 4 cols bf16 = 64 bytes
//   v0 = A[0:2][k:k+4], v1 = A[2:4][k:k+4]
//   v2 = A[4:6][k:k+4], v3 = A[6:8][k:k+4]
// B 加载: 4 regs = 4 rows × 8 cols bf16 = 64 bytes
//   v4 = B[k:k+4][0:2], v5 = B[k:k+4][2:4]
//   v6 = B[k:k+4][4:6], v7 = B[k:k+4][6:8]

	.macro LOAD_A0_B0_8
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_8
	// Load next A and B (prefetch)
	ldp q8,  q25, [A_ADDR], #32
	ldp q26, q27, [A_ADDR], #32
	ldp q28, q29, [B_ADDR_1], #32
	ldp q30, q31, [B_ADDR_1], #32

	// Compute 16 bfmmla: A0×B0 (v0-v3 × v4-v7)
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h

	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h

	bfmmla v17.4s, v2.8h, v4.8h
	bfmmla v18.4s, v2.8h, v5.8h
	bfmmla v19.4s, v2.8h, v6.8h
	bfmmla v20.4s, v2.8h, v7.8h

	bfmmla v21.4s, v3.8h, v4.8h
	bfmmla v22.4s, v3.8h, v5.8h
	bfmmla v23.4s, v3.8h, v6.8h
	bfmmla v24.4s, v3.8h, v7.8h
	.endm

	.macro COMPUTE_A2_B2_8
	// Load next A and B (prefetch)
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32

	// Compute 16 bfmmla: A1×B1 (v8,v25-v27 × v28-v31)
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h

	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h

	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

	.macro COMPUTE_A3_B3_8
	// Compute with last A,B pair (no prefetch)
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h

	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h

	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

// ==================== STRIDED PATH MACROS ====================
// 从原始 A 跨 stride (LDA_B) 读取，写入 A_reorder（首个 N iteration 使用）
// A 每行加载 4 bf16 = 8 bytes → 8 行 = 64 bytes = 4 NEON regs

	.macro LOAD_A0_B0_STRIDED_8
	// ── 从原始 A 跨 stride 加载 A0 → v0-v3 ──
	// 预计算 8 个行地址，row 0 直接用 A_ADDR，最大深度 3
	add  x11, A_ADDR, LDA_B                // 1*LDA
	add  x12, A_ADDR, LDA_B, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA_B, lsl #2        // 4*LDA
	sub  x13, x14, LDA_B                   // 3*LDA = 4-1
	add  x15, x11, LDA_B, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA_B, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA_B, lsl #1           // 7*LDA = 5+2

	// 8 路独立发射，无寄存器依赖
	// 每行加载 4 bf16 = 8 bytes = 1 doubleword
	ld1  {v0.d}[0], [A_ADDR]           // row 0 → v0.d[0]
	ld1  {v0.d}[1], [x11]              // row 1 → v0.d[1]
	ld1  {v1.d}[0], [x12]              // row 2 → v1.d[0]
	ld1  {v1.d}[1], [x13]              // row 3 → v1.d[1]
	ld1  {v2.d}[0], [x14]              // row 4 → v2.d[0]
	ld1  {v2.d}[1], [x15]              // row 5 → v2.d[1]
	ld1  {v3.d}[0], [x16]              // row 6 → v3.d[0]
	ld1  {v3.d}[1], [x17]              // row 7 → v3.d[1]

	// ── 写入 A_reorder（连续 64 字节）──
	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	// ── A_ADDR 前进到下一个 K block（原始 A 中 +4 列 = +8 bytes）──
	add  A_ADDR, A_ADDR, #8

	// ── 加载 B0（同 cached path）──
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED_8
	// ── 从原始 A 跨 stride 加载 A1 → v8, v25-v27 ──
	add  x11, A_ADDR, LDA_B                // 1*LDA
	add  x12, A_ADDR, LDA_B, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA_B, lsl #2        // 4*LDA
	sub  x13, x14, LDA_B                   // 3*LDA = 4-1
	add  x15, x11, LDA_B, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA_B, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA_B, lsl #1           // 7*LDA = 5+2

	ld1  {v8.d}[0],  [A_ADDR]          // row 0 → v8.d[0]
	ld1  {v8.d}[1],  [x11]             // row 1 → v8.d[1]
	ld1  {v25.d}[0], [x12]             // row 2 → v25.d[0]
	ld1  {v25.d}[1], [x13]             // row 3 → v25.d[1]
	ld1  {v26.d}[0], [x14]             // row 4 → v26.d[0]
	ld1  {v26.d}[1], [x15]             // row 5 → v26.d[1]
	ld1  {v27.d}[0], [x16]             // row 6 → v27.d[0]
	ld1  {v27.d}[1], [x17]             // row 7 → v27.d[1]

	// ── 写入 A_reorder ──
	stp  q8,  q25, [A_REORDER], #32
	stp  q26, q27, [A_REORDER], #32

	// ── A_ADDR 前进到下一个 K block ──
	add  A_ADDR, A_ADDR, #8

	// ── 加载 B1 ──
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	// ── 计算 A0 × B0 ──
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h

	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h

	bfmmla v17.4s, v2.8h, v4.8h
	bfmmla v18.4s, v2.8h, v5.8h
	bfmmla v19.4s, v2.8h, v6.8h
	bfmmla v20.4s, v2.8h, v7.8h

	bfmmla v21.4s, v3.8h, v4.8h
	bfmmla v22.4s, v3.8h, v5.8h
	bfmmla v23.4s, v3.8h, v6.8h
	bfmmla v24.4s, v3.8h, v7.8h
	.endm

	.macro COMPUTE_A2_B2_STRIDED_8
	// ── 从原始 A 跨 stride 加载 A2 → v0-v3 ──
	add  x11, A_ADDR, LDA_B                // 1*LDA
	add  x12, A_ADDR, LDA_B, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA_B, lsl #2        // 4*LDA
	sub  x13, x14, LDA_B                   // 3*LDA = 4-1
	add  x15, x11, LDA_B, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA_B, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA_B, lsl #1           // 7*LDA = 5+2

	ld1  {v0.d}[0], [A_ADDR]           // row 0 → v0.d[0]
	ld1  {v0.d}[1], [x11]              // row 1 → v0.d[1]
	ld1  {v1.d}[0], [x12]              // row 2 → v1.d[0]
	ld1  {v1.d}[1], [x13]              // row 3 → v1.d[1]
	ld1  {v2.d}[0], [x14]              // row 4 → v2.d[0]
	ld1  {v2.d}[1], [x15]              // row 5 → v2.d[1]
	ld1  {v3.d}[0], [x16]              // row 6 → v3.d[0]
	ld1  {v3.d}[1], [x17]              // row 7 → v3.d[1]

	// ── 写入 A_reorder ──
	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	// ── A_ADDR 前进到下一个 K block ──
	add  A_ADDR, A_ADDR, #8

	// ── 加载 B2 ──
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	// ── 计算 A1 × B1 ──
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h

	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h

	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

.macro GEMM_BODY func_name, mrows, cmpsfx, load_c, store_c, ldc_shift, c_adv, a_mode=repack

#ifdef __APPLE__
	.global _\func_name
_\func_name:
#else
	.global \func_name
\func_name:
#endif
	INIT

	// New calling convention: x3 = A_REORDER, x4 = PARAMS
	mov  A_REORDER, x3              // save A_REORDER → x6

	// Load m,k,n,lda,ldb,ldc from params struct (6×int32 = 24 bytes)
	mov  x11, x4                    // keep params pointer before K overwrites x4
	ldp  w3, w4, [x11]              // M, K
	ldp  w5, w7, [x11, #8]          // N, LDA
	ldp  w8, w12, [x11, #16]        // LDB, LDC

	lsl  LDC_B, x12, #\ldc_shift

	lsl  LDA_B, LDA_B, #1        // LDA_B = lda * 2 (bytes)

	mov  A_ORIG_M, A_ORIG
	mov  C_ROW_BASE, C_ADDR_0

	M_LOOP_BEGIN_\mrows

	.ifc \a_mode,packed
	mov  A_REORDER_M, A_ORIG_M
	.else
	mov  A_REORDER_M, A_REORDER
	mov  FIRST_N, #1
	.endif
	mov  C_N_BASE, C_ROW_BASE
	mov  B_N_BASE, B_ADDR

.L\func_name\()_n_loop:
	asr  N_COUNT, N, #3
	cmp  N_COUNT, #0
	b.eq .L\func_name\()_end

.L\func_name\()_n_iter:
	mov  B_ADDR_1, B_N_BASE
	mov  C_M_BASE, C_N_BASE

	.ifc \a_mode,repack
	cmp  FIRST_N, #1
	b.eq .L\func_name\()_strided
	.endif

// ── CACHED PATH ───────────────────────────────────────────────────
	mov  A_ADDR, A_REORDER_M

	\load_c

	asr  K_COUNT, K, #3
	cmp  K_COUNT, #0
	b.eq .L\func_name\()_end
	subs K_COUNT, K_COUNT, #1

	LOAD_A0_B0_\cmpsfx

	b.eq .L\func_name\()_cached_tail

.L\func_name\()_cached_loop:
	COMPUTE_A1_B1_\cmpsfx
	COMPUTE_A2_B2_\cmpsfx
	subs K_COUNT, K_COUNT, #1
	b.gt .L\func_name\()_cached_loop

.L\func_name\()_cached_tail:
	COMPUTE_A1_B1_\cmpsfx
	COMPUTE_A3_B3_\cmpsfx

	b .L\func_name\()_epilogue

// ── STRIDED PATH ──────────────────────────────────────────────────
	.ifc \a_mode,repack
.L\func_name\()_strided:
	mov  A_ADDR, A_ORIG_M

	\load_c

	asr  K_COUNT, K, #3
	cmp  K_COUNT, #0
	b.eq .L\func_name\()_end
	subs K_COUNT, K_COUNT, #1

	LOAD_A0_B0_STRIDED_\mrows

	b.eq .L\func_name\()_strided_tail

.L\func_name\()_strided_loop:
	COMPUTE_A1_B1_STRIDED_\mrows
	COMPUTE_A2_B2_STRIDED_\mrows
	subs K_COUNT, K_COUNT, #1
	b.gt .L\func_name\()_strided_loop

.L\func_name\()_strided_tail:
	COMPUTE_A1_B1_STRIDED_\mrows
	COMPUTE_A3_B3_\cmpsfx

	mov  FIRST_N, #0
	.endif

// ── EPILOGUE ──────────────────────────────────────────────────────
.L\func_name\()_epilogue:
	\store_c
	add  C_N_BASE, C_N_BASE, #\c_adv
	add  B_N_BASE, B_N_BASE, LDB, lsl #4

	subs N_COUNT, N_COUNT, #1
	b.gt .L\func_name\()_n_iter

	M_LOOP_END_\mrows

.L\func_name\()_end:
	RESTORE
	ret

.endm   // GEMM_BODY

// ==================== 主入口 (m=8) — GEMM_BODY 宏生成 ====================

// f32 store (m=8)

// bf16 store (m=8)

// Tail kernels (m=1,2,4) — appended from bf16gemm_k_tail.S
// Note: INIT/RESTORE macros reused from main kernel above
// ═══════════════════════════════════════════════════════════════


// ════════════════════════════════════════════════════════════════════
// LOAD_C / STORE_C  按行数特化
// bf16 使用 bfmmla，累加寄存器格式与 i8 的 smmla 一致（zip1/zip2 2d）
// ════════════════════════════════════════════════════════════════════

// ── m=1/2：只涉及 row0/row1 → v9~v12 ────────────────────────────
// C 布局 [col0..3 | col4..7]，每行 8×float32 = 32 bytes
//   v9  = [r0c0,r0c1,r1c0,r1c1]
//   v10 = [r0c2,r0c3,r1c2,r1c3]
//   v11 = [r0c4,r0c5,r1c4,r1c5]
//   v12 = [r0c6,r0c7,r1c6,r1c7]

.macro LOAD_C_2
	// 读 row0, row1
	add  x11, C_M_BASE, LDC_B          // row1
	ldp  q0, q1, [C_M_BASE]            // row0: col0-3, col4-7
	ldp  q2, q3, [x11]                 // row1: col0-3, col4-7
	ZIP12_D v9,  v10, v0, v2
	ZIP12_D v11, v12, v1, v3
.endm

.macro STORE_C_2
	ZIP12_D v0, v2, v9,  v10
	ZIP12_D v1, v3, v11, v12
	add  x11, C_M_BASE, LDC_B
	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
.endm

// m=1 时 row1 是无效填充行，只写 row0

.macro LOAD_C_1
	// 只读 row0；row1 的寄存器置零（供 bfmmla 正确填充 r1 槽位，结果不写回）
	ldp  q0, q1, [C_M_BASE]            // row0: col0-3, col4-7
	movi v2.16b, #0                    // row1 填零（虚拟行）
	movi v3.16b, #0
	ZIP12_D v9,  v10, v0, v2
	ZIP12_D v11, v12, v1, v3
.endm

.macro STORE_C_1
	// 只写 row0
	zip1 v0.2d, v9.2d, v10.2d          // row0: col0-3
	zip1 v1.2d, v11.2d, v12.2d         // row0: col4-7
	stp  q0, q1, [C_M_BASE]
.endm

// ── m=4：row0~row3 → v9~v16 ─────────────────────────────────────
//   v9~v12 : row0/row1（同 m=2）
//   v13~v16: row2/row3
.macro LOAD_C_4
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B               // 3*LDC

	ldp  q0, q1, [C_M_BASE]
	ldp  q2, q3, [x11]
	ldp  q4, q5, [x12]
	ldp  q6, q7, [x13]

	ZIP12_D v9,  v10, v0, v2
	ZIP12_D v11, v12, v1, v3

	ZIP12_D v13, v14, v4, v6
	ZIP12_D v15, v16, v5, v7
.endm

.macro STORE_C_4
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B

	ZIP12_D v0, v2, v9,  v10
	ZIP12_D v1, v3, v11, v12

	ZIP12_D v4, v6, v13, v14
	ZIP12_D v5, v7, v15, v16

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
.endm

// ── LOAD_C_B 系列：bf16→f32 转换 ─────────────────────────────────
	.macro LOAD_C_1_B
	LOAD_BF16_ROW_F32 C_M_BASE, v2, v3
	movi v4.16b, #0
	movi v5.16b, #0
	ZIP12_D v9,  v10, v2, v4
	ZIP12_D v11, v12, v3, v5
	.endm
	.macro LOAD_C_2_B
	add  x11, C_M_BASE, LDC_B
	LOAD_BF16_ROW_F32 C_M_BASE, v2, v3
	LOAD_BF16_ROW_F32 x11, v4, v5
	ZIP12_D v9,  v10, v2, v4
	ZIP12_D v11, v12, v3, v5
	.endm
	.macro LOAD_C_4_B
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	LOAD_BF16_ROW_F32 C_M_BASE, v2, v3
	LOAD_BF16_ROW_F32 x11, v4, v5
	ZIP12_D v9,  v10, v2, v4
	ZIP12_D v11, v12, v3, v5
	LOAD_BF16_ROW_F32 x12, v2, v3
	LOAD_BF16_ROW_F32 x13, v4, v5
	ZIP12_D v13, v14, v2, v4
	ZIP12_D v15, v16, v3, v5
	.endm

// ── STORE_C_B 系列：f32→bf16 转换 ─────────────────────────────────
	.macro STORE_C_1_B
	zip1 v0.2d, v9.2d, v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	bfcvtn  v2.4h, v0.4s
	bfcvtn2 v2.8h, v1.4s
	str  q2, [C_M_BASE]
	.endm
	.macro STORE_C_2_B
	ZIP12_D v0, v2, v9,  v10
	ZIP12_D v1, v3, v11, v12
	bfcvtn  v4.4h, v0.4s
	bfcvtn2 v4.8h, v1.4s
	bfcvtn  v5.4h, v2.4s
	bfcvtn2 v5.8h, v3.4s
	add  x11, C_M_BASE, LDC_B
	str  q4, [C_M_BASE]
	str  q5, [x11]
	.endm
	.macro STORE_C_4_B
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	ZIP12_D v0, v2, v9,  v10
	ZIP12_D v1, v3, v11, v12
	ZIP12_D v4, v6, v13, v14
	ZIP12_D v5, v7, v15, v16
	bfcvtn  v25.4h, v0.4s
	bfcvtn2 v25.8h, v1.4s
	bfcvtn  v26.4h, v2.4s
	bfcvtn2 v26.8h, v3.4s
	bfcvtn  v27.4h, v4.4s
	bfcvtn2 v27.8h, v5.4s
	bfcvtn  v28.4h, v6.4s
	bfcvtn2 v28.8h, v7.4s
	str  q25, [C_M_BASE]
	str  q26, [x11]
	str  q27, [x12]
	str  q28, [x13]
	.endm

// ════════════════════════════════════════════════════════════════════
// CACHED PATH  按 mrows 特化
// 从 A_reorder 连续读取（后续 N iteration 使用）
// A_reorder 中数据已经连续，每 K-block：
//   m=1/2 : 16 bytes（1个 q）
//   m=4   : 32 bytes（2个 q）
// 每次处理 2 个 K-block（= 32 bytes A / m=1/2，64 bytes / m=4）
// B 侧与 m=8 主内核完全相同：每 K-block 4 个 q (64 bytes)
// ════════════════════════════════════════════════════════════════════

// ── m=1/2 cached：A 用 v0,v8；B 与 m=8 相同 ──────────────────────

.macro LOAD_A0_B0_2
	ldr  q0, [A_ADDR], #16             // A block0: row0|row1 (or row0|zero)
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1_2
	ldr  q8, [A_ADDR], #16             // A block1
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h
.endm

.macro COMPUTE_A2_B2_2
	ldr  q0, [A_ADDR], #16
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
.endm

.macro COMPUTE_A3_B3_2
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
.endm

// ── m=4 cached：A 用 v0~v1, v8,v25；B 与 m=8 相同 ────────────────

.macro LOAD_A0_B0_4
	ldp  q0, q1, [A_ADDR], #32         // A block0: row0|row1, row2|row3
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1_4
	ldp  q8, q25, [A_ADDR], #32        // A block1
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h

	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h
.endm

.macro COMPUTE_A2_B2_4
	ldp  q0, q1, [A_ADDR], #32
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	bfmmla v9.4s,  v8.8h,  v28.8h
	bfmmla v10.4s, v8.8h,  v29.8h
	bfmmla v11.4s, v8.8h,  v30.8h
	bfmmla v12.4s, v8.8h,  v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h
.endm

.macro COMPUTE_A3_B3_4
	bfmmla v9.4s,  v8.8h,  v28.8h
	bfmmla v10.4s, v8.8h,  v29.8h
	bfmmla v11.4s, v8.8h,  v30.8h
	bfmmla v12.4s, v8.8h,  v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h
.endm

// ════════════════════════════════════════════════════════════════════
// STRIDED PATH  按 mrows 特化
// 从原始 A 跨 LDA_B 读取，同时写 A_reorder（首个 N iteration 使用）
// ════════════════════════════════════════════════════════════════════

// ── m=1 strided：只读 row0，row1 槽填零 ─────────────────────────

.macro LOAD_A0_B0_STRIDED_1
	ldr  d0, [A_ADDR]                  // row0 → v0.d[0] (ldr d zero-extends to 128b)
	str  q0, [A_REORDER], #16          // 写 A_reorder (upper 64b=0)
	add  A_ADDR, A_ADDR, #8            // 前进到下一 K-block

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1_STRIDED_1
	ldr  d8, [A_ADDR]
	str  q8, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h
.endm

.macro COMPUTE_A2_B2_STRIDED_1
	ldr  d0, [A_ADDR]
	mov v0.d[1], xzr
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
.endm

// ── m=2 strided：读 row0/row1，各取 4 bf16 拼成一个 q ────────────

.macro LOAD_A0_B0_STRIDED_2
	add  x11, A_ADDR, LDA_B
	ldr  d0, [A_ADDR]                  // row0 → v0.d[0]
	ld1  {v0.d}[1], [x11]             // row1 → v0.d[1]
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1_STRIDED_2
	add  x11, A_ADDR, LDA_B
	ldr  d8, [A_ADDR]
	ld1  {v8.d}[1], [x11]
	str  q8, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h
.endm

.macro COMPUTE_A2_B2_STRIDED_2
	add  x11, A_ADDR, LDA_B
	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
.endm

// ── m=4 strided：读 row0~row3，打包成 2 个 q ────────────────────

.macro LOAD_A0_B0_STRIDED_4
	add  x11, A_ADDR, LDA_B
	add  x12, A_ADDR, LDA_B, lsl #1
	add  x14, A_ADDR, LDA_B, lsl #2
	sub  x13, x14, LDA_B               // 3*LDA

	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]             // v0 = row0|row1
	ldr  d1, [x12]
	ld1  {v1.d}[1], [x13]             // v1 = row2|row3

	stp  q0, q1, [A_REORDER], #32
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1_STRIDED_4
	add  x11, A_ADDR, LDA_B
	add  x12, A_ADDR, LDA_B, lsl #1
	add  x14, A_ADDR, LDA_B, lsl #2
	sub  x13, x14, LDA_B

	ldr  d8, [A_ADDR]
	ld1  {v8.d}[1], [x11]
	ldr  d25, [x12]
	ld1  {v25.d}[1], [x13]

	stp  q8, q25, [A_REORDER], #32
	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h

	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h
.endm

.macro COMPUTE_A2_B2_STRIDED_4
	add  x11, A_ADDR, LDA_B
	add  x12, A_ADDR, LDA_B, lsl #1
	add  x14, A_ADDR, LDA_B, lsl #2
	sub  x13, x14, LDA_B

	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ldr  d1, [x12]
	ld1  {v1.d}[1], [x13]

	stp  q0, q1, [A_REORDER], #32
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	bfmmla v9.4s,  v8.8h,  v28.8h
	bfmmla v10.4s, v8.8h,  v29.8h
	bfmmla v11.4s, v8.8h,  v30.8h
	bfmmla v12.4s, v8.8h,  v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h
.endm

// ════════════════════════════════════════════════════════════════════
// LOAD_ZERO — zero-initialise accumulators (no C read)
// For use with GEMM_BODY to skip the C-load phase entirely.
// ════════════════════════════════════════════════════════════════════
	.macro LOAD_ZERO_8
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	movi v13.4s, #0
	movi v14.4s, #0
	movi v15.4s, #0
	movi v16.4s, #0
	movi v17.4s, #0
	movi v18.4s, #0
	movi v19.4s, #0
	movi v20.4s, #0
	movi v21.4s, #0
	movi v22.4s, #0
	movi v23.4s, #0
	movi v24.4s, #0
	.endm

	.macro LOAD_ZERO_2
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	.endm

	.macro LOAD_ZERO_1
	LOAD_ZERO_2
	.endm

	.macro LOAD_ZERO_4
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	movi v13.4s, #0
	movi v14.4s, #0
	movi v15.4s, #0
	movi v16.4s, #0
	.endm

// ════════════════════════════════════════════════════════════════════
// GEMM_BODY：函数主体宏
//   \func_name : 符号名（不含平台前缀，由宏内部处理）
//   \mrows     : 1 / 2 / 4
// ════════════════════════════════════════════════════════════════════


// f32 store

// bf16 store (_b suffix)
// bf16 store, no C load (_nld_b suffix) — accumulators zero-init

// ── Fused w13 + gate*up (Task 2: linear, no silu). bf16 out, no C load. ──
// N iterates the full 2*F_pad packed columns; each 8-col tile stores 4 output
// features (8 bytes), so c_adv=8; ldc_shift=1 (bf16 row stride = F_pad*2).

// ── Fused w13 + SiLU-and-mul (poly4/5/6 exp). bf16 out, no C load, m=8. ──

// M-tail (m=1/2/4) fused kernels — cmpsfx 2 for m1/m2, 4 for m4 (as ld1/2/4).

// Pre-packed-A variants (a_mode=packed): A already in reorder layout,
// shared across an N-split team; skip the in-kernel strided repack.

// ── Constant table for the fused SiLU epilogue (broadcast-loaded via ld1r) ──
	.balign 16
Csilu_poly_consts:
	.float 1.4426950408889634   // invln2
	.float 0.6931471805599453   // ln2
	.float 0.16666666           // c3
	.float 0.04166666           // c4
	.float 0.00833333           // c5
	.float 0.0013888889         // c6
	.float 87.0                 // clamp_hi
	.float -87.0                // clamp_lo


// ==================== packed-C m8 fused-silu kernels (fused_cpp) ==========
// packc store + c_adv=64 (one reorder K-block) + packed A read.
// ==================== PACKED-READ REORDER-M8 TAIL MACROS ====================
// Read only the first mr rows from a reorder-m8 8-row K-block (64B), skipping
// padding rows, then advance A_ADDR by the full 64B block. bfmmla A operand =
// 2 rows x 4 cols = 16B: v0=rows0,1; v1=rows2,3. P2 reads 16B (rows0,1); P4
// reads 32B (rows0-3). B side identical to m8. Used by the packed-read tail
// kernels so gather's pre-packed A is consumed with no in-kernel repack.

// -- P2: rows 0,1 (m1 stores row0 only; m2 stores rows0,1) --
.macro LOAD_A0_B0_P2
	ldr  q0, [A_ADDR], #64
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm
.macro COMPUTE_A1_B1_P2
	ldr  q8, [A_ADDR], #64
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h
.endm
.macro COMPUTE_A2_B2_P2
	ldr  q0, [A_ADDR], #64
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
.endm
.macro COMPUTE_A3_B3_P2
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
.endm

// -- P4: rows 0-3 --
.macro LOAD_A0_B0_P4
	ldp  q0, q1, [A_ADDR], #64
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
.endm
.macro COMPUTE_A1_B1_P4
	ldp  q8, q25, [A_ADDR], #64
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h
	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h
.endm
.macro COMPUTE_A2_B2_P4
	ldp  q0, q1, [A_ADDR], #64
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	bfmmla v9.4s,  v8.8h,  v28.8h
	bfmmla v10.4s, v8.8h,  v29.8h
	bfmmla v11.4s, v8.8h,  v30.8h
	bfmmla v12.4s, v8.8h,  v31.8h
	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h
.endm
.macro COMPUTE_A3_B3_P4
	bfmmla v9.4s,  v8.8h,  v28.8h
	bfmmla v10.4s, v8.8h,  v29.8h
	bfmmla v11.4s, v8.8h,  v30.8h
	bfmmla v12.4s, v8.8h,  v31.8h
	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h
.endm

// -- packc tail stores: first mr rows written contiguous at C_M_BASE (row r at
//    +r*8), mirroring STORE_C_SILU_POLY_PACKC_8 truncated to mr rows. --
	.macro STORE_C_SILU_POLY_PACKC_M4 deg
	SILU_CONSTS_LOAD
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE]
	ZIP12_D v0, v1, v13, v14
	ZIP12_D v2, v3, v15, v16
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE, #16]
	.endm
	.macro STORE_C_SILU_POLY_PACKC_M2 deg
	SILU_CONSTS_LOAD
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	SILU_ROW 1, 3, 4, 5, 6, 9, \deg
	ins  v0.d[1], v1.d[0]
	str  q0, [C_M_BASE]
	.endm
	.macro STORE_C_SILU_POLY_PACKC_M1 deg
	SILU_CONSTS_LOAD
	ZIP12_D v0, v1, v9,  v10
	ZIP12_D v2, v3, v11, v12
	SILU_ROW 0, 2, 4, 5, 6, 9, \deg
	str  d0, [C_M_BASE]
	.endm
	.macro STORE_C_SILU_POLY4_PACKC_M4
	STORE_C_SILU_POLY_PACKC_M4 4
	.endm
	.macro STORE_C_SILU_POLY5_PACKC_M4
	STORE_C_SILU_POLY_PACKC_M4 5
	.endm
	.macro STORE_C_SILU_POLY6_PACKC_M4
	STORE_C_SILU_POLY_PACKC_M4 6
	.endm
	.macro STORE_C_SILU_POLY4_PACKC_M2
	STORE_C_SILU_POLY_PACKC_M2 4
	.endm
	.macro STORE_C_SILU_POLY5_PACKC_M2
	STORE_C_SILU_POLY_PACKC_M2 5
	.endm
	.macro STORE_C_SILU_POLY6_PACKC_M2
	STORE_C_SILU_POLY_PACKC_M2 6
	.endm
	.macro STORE_C_SILU_POLY4_PACKC_M1
	STORE_C_SILU_POLY_PACKC_M1 4
	.endm
	.macro STORE_C_SILU_POLY5_PACKC_M1
	STORE_C_SILU_POLY_PACKC_M1 5
	.endm
	.macro STORE_C_SILU_POLY6_PACKC_M1
	STORE_C_SILU_POLY_PACKC_M1 6
	.endm

GEMM_BODY bf16gemm_k_ldp_silu_poly4_packc, 8, 8, LOAD_ZERO_8, STORE_C_SILU_POLY4_PACKC_8, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly5_packc, 8, 8, LOAD_ZERO_8, STORE_C_SILU_POLY5_PACKC_8, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly6_packc, 8, 8, LOAD_ZERO_8, STORE_C_SILU_POLY6_PACKC_8, 1, 64, packed

// w13 fused-silu packed-read reorder-m8 TAIL kernels (packc store, no repack)
GEMM_BODY bf16gemm_k_ldp_silu_poly4_packc_m4, 4, P4, LOAD_ZERO_4, STORE_C_SILU_POLY4_PACKC_M4, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly5_packc_m4, 4, P4, LOAD_ZERO_4, STORE_C_SILU_POLY5_PACKC_M4, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly6_packc_m4, 4, P4, LOAD_ZERO_4, STORE_C_SILU_POLY6_PACKC_M4, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly4_packc_m2, 2, P2, LOAD_ZERO_2, STORE_C_SILU_POLY4_PACKC_M2, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly5_packc_m2, 2, P2, LOAD_ZERO_2, STORE_C_SILU_POLY5_PACKC_M2, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly6_packc_m2, 2, P2, LOAD_ZERO_2, STORE_C_SILU_POLY6_PACKC_M2, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly4_packc_m1, 1, P2, LOAD_ZERO_1, STORE_C_SILU_POLY4_PACKC_M1, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly5_packc_m1, 1, P2, LOAD_ZERO_1, STORE_C_SILU_POLY5_PACKC_M1, 1, 64, packed
GEMM_BODY bf16gemm_k_ldp_silu_poly6_packc_m1, 1, P2, LOAD_ZERO_1, STORE_C_SILU_POLY6_PACKC_M1, 1, 64, packed


// ==================== packed-read plain fp32 GEMM (w2) ====================
// Reads reorder-m8 pre-packed A (the packed intermediate written by the packc
// w13 store), computes A x B^T -> C row-major fp32. Same as bf16gemm_k_ld but
// a_mode=packed (no in-kernel strided repack). Used by the fused-packa w2
// stage. M padded to a multiple of 8.
GEMM_BODY bf16gemm_k_ldp, 8, 8, LOAD_ZERO_MAIN_8, STORE_C_ROWMAJOR, 2, 32, packed

// w2 plain fp32 packed-read reorder-m8 TAIL kernels (rowmajor store, no repack)
GEMM_BODY bf16gemm_k_ldp_m4, 4, P4, LOAD_ZERO_4, STORE_C_4, 2, 32, packed
GEMM_BODY bf16gemm_k_ldp_m2, 2, P2, LOAD_ZERO_2, STORE_C_2, 2, 32, packed
GEMM_BODY bf16gemm_k_ldp_m1, 1, P2, LOAD_ZERO_1, STORE_C_1, 2, 32, packed
