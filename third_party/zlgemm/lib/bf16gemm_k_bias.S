// bf16gemm_k_bias.S — bf16 GEMM kernel with fp32 bias addition (no C load)
// void bf16gemm_k_ld_bias_f(const uint16_t *A,            // x0
//                           const uint16_t *B_reordered,  // x1
//                           float          *C,            // x2  fp32 output
//                           uint16_t       *A_reorder,    // x3  A reorder buffer
//                           const gemm_params_t *params,  // x4  {m,k,n,lda,ldb,ldc}
//                           const float    *bias);        // x5  bias[N] fp32
//
// Key design:
//   - Accumulators zero-initialised (no C read)
//   - bfmmla bf16×bf16 → fp32 accumulate
//   - STORE: zip (blocked→row-major) → ldp bias → fadd → stp
//   - NO scvtf needed — accumulators are already fp32 (unlike i8 int32)
//   - Bias is per-column, shared across rows; loaded once per N-block (8 values)

	.text
	.align 4

// ── Register aliases ──────────────────────────────────
#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA_B       x7    // A row stride in bytes (= lda * 2)
#define LDC_B       x9    // C row stride in bytes = ldc * 4
#define LDB         x8    // K_big of packed B
#define BIAS_ADDR   x21   // bias pointer, callee-saved

#define N_COUNT     x30
#define C_N_BASE    x27
#define M_COUNT     x10
#define B_ADDR_1    x19
#define B_N_BASE    x22
#define K_COUNT     x23
#define FIRST_N     x24
#define C_ROW_BASE  x25
#define C_M_BASE    x26
#define A_REORDER_M x28
#define A_ADDR      x29
#define A_ORIG_M    x20

// ── M_LOOP macros ─────────────────────────────────────
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

// ── INIT / RESTORE ────────────────────────────────────
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

// ═══════════════════════════════════════════════════════
// LOAD_ZERO — zero-initialise fp32 accumulators (no C read)
// ═══════════════════════════════════════════════════════

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
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
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

// ═══════════════════════════════════════════════════════
// STORE_C_*_BIAS_F — store fp32 with bias addition
// Accumulators are already fp32 → just zip, load bias, fadd, stp
// ═══════════════════════════════════════════════════════

// ── m=8 ───────────────────────────────────────────────
	.macro STORE_C_ROWMAJOR_BIAS_F
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	add  x15, x11, LDC_B, lsl #2
	add  x16, x12, LDC_B, lsl #2
	add  x17, x15, LDC_B, lsl #1

	// zip fp32 accumulators → row-major
	zip1 v0.2d, v9.2d,  v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	zip2 v2.2d, v9.2d,  v10.2d
	zip2 v3.2d, v11.2d, v12.2d

	zip1 v4.2d, v13.2d, v14.2d
	zip1 v5.2d, v15.2d, v16.2d
	zip2 v6.2d, v13.2d, v14.2d
	zip2 v7.2d, v15.2d, v16.2d

	zip1 v8.2d, v17.2d, v18.2d
	zip1 v25.2d, v19.2d, v20.2d
	zip2 v26.2d, v17.2d, v18.2d
	zip2 v27.2d, v19.2d, v20.2d

	zip1 v28.2d, v21.2d, v22.2d
	zip1 v29.2d, v23.2d, v24.2d
	zip2 v30.2d, v21.2d, v22.2d
	zip2 v31.2d, v23.2d, v24.2d

	// load bias: v16=[b0,b1,b2,b3], v17=[b4,b5,b6,b7]
	// (v16,v17 were accumulator regs, now dead after zip)
	ldp  q16, q17, [BIAS_ADDR]

	// fadd bias to each row
	fadd v0.4s, v0.4s, v16.4s
	fadd v1.4s, v1.4s, v17.4s
	fadd v2.4s, v2.4s, v16.4s
	fadd v3.4s, v3.4s, v17.4s
	fadd v4.4s, v4.4s, v16.4s
	fadd v5.4s, v5.4s, v17.4s
	fadd v6.4s, v6.4s, v16.4s
	fadd v7.4s, v7.4s, v17.4s
	fadd v8.4s, v8.4s, v16.4s
	fadd v25.4s, v25.4s, v17.4s
	fadd v26.4s, v26.4s, v16.4s
	fadd v27.4s, v27.4s, v17.4s
	fadd v28.4s, v28.4s, v16.4s
	fadd v29.4s, v29.4s, v17.4s
	fadd v30.4s, v30.4s, v16.4s
	fadd v31.4s, v31.4s, v17.4s

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	stp  q8, q25, [x14]
	stp  q26, q27, [x15]
	stp  q28, q29, [x16]
	stp  q30, q31, [x17]
	.endm

// ── m=1 ───────────────────────────────────────────────
	.macro STORE_C_1_BIAS_F
	zip1 v0.2d, v9.2d,  v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	ldp  q2, q3, [BIAS_ADDR]
	fadd v0.4s, v0.4s, v2.4s
	fadd v1.4s, v1.4s, v3.4s
	stp  q0, q1, [C_M_BASE]
	.endm

// ── m=2 ───────────────────────────────────────────────
	.macro STORE_C_2_BIAS_F
	zip1 v0.2d, v9.2d,  v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	zip2 v2.2d, v9.2d,  v10.2d
	zip2 v3.2d, v11.2d, v12.2d
	ldp  q4, q5, [BIAS_ADDR]
	fadd v0.4s, v0.4s, v4.4s
	fadd v1.4s, v1.4s, v5.4s
	fadd v2.4s, v2.4s, v4.4s
	fadd v3.4s, v3.4s, v5.4s
	add  x11, C_M_BASE, LDC_B
	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	.endm

// ── m=4 ───────────────────────────────────────────────
	.macro STORE_C_4_BIAS_F
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B

	zip1 v0.2d, v9.2d,  v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	zip2 v2.2d, v9.2d,  v10.2d
	zip2 v3.2d, v11.2d, v12.2d

	zip1 v4.2d, v13.2d, v14.2d
	zip1 v5.2d, v15.2d, v16.2d
	zip2 v6.2d, v13.2d, v14.2d
	zip2 v7.2d, v15.2d, v16.2d

	ldp  q16, q17, [BIAS_ADDR]
	fadd v0.4s, v0.4s, v16.4s
	fadd v1.4s, v1.4s, v17.4s
	fadd v2.4s, v2.4s, v16.4s
	fadd v3.4s, v3.4s, v17.4s
	fadd v4.4s, v4.4s, v16.4s
	fadd v5.4s, v5.4s, v17.4s
	fadd v6.4s, v6.4s, v16.4s
	fadd v7.4s, v7.4s, v17.4s

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	.endm

// ═══════════════════════════════════════════════════════
// CACHED PATH  (m=8, m=1/2, m=4)
// Identical to bf16gemm_k.S — bfmmla is the same
// ═══════════════════════════════════════════════════════

// ── m=8 cached ───────────────────────────────────────
	.macro LOAD_A0_B0_8
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_8
	ldp q8,  q25, [A_ADDR], #32
	ldp q26, q27, [A_ADDR], #32
	ldp q28, q29, [B_ADDR_1], #32
	ldp q30, q31, [B_ADDR_1], #32

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
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32

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

// ── m=1/2 cached ──────────────────────────────────────
	.macro LOAD_A0_B0_2
	ldr  q0, [A_ADDR], #16
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_2
	ldr  q8, [A_ADDR], #16
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

// ── m=4 cached ────────────────────────────────────────
	.macro LOAD_A0_B0_4
	ldp  q0, q1, [A_ADDR], #32
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_4
	ldp  q8, q25, [A_ADDR], #32
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

// ═══════════════════════════════════════════════════════
// STRIDED PATH (m=8, m=1, m=2, m=4)
// Identical to bf16gemm_k.S
// ═══════════════════════════════════════════════════════

// ── m=8 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_8
	add  x11, A_ADDR, LDA_B
	add  x12, A_ADDR, LDA_B, lsl #1
	add  x14, A_ADDR, LDA_B, lsl #2
	sub  x13, x14, LDA_B
	add  x15, x11, LDA_B, lsl #2
	add  x16, x12, LDA_B, lsl #2
	add  x17, x15, LDA_B, lsl #1

	ld1  {v0.d}[0], [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ld1  {v1.d}[0], [x12]
	ld1  {v1.d}[1], [x13]
	ld1  {v2.d}[0], [x14]
	ld1  {v2.d}[1], [x15]
	ld1  {v3.d}[0], [x16]
	ld1  {v3.d}[1], [x17]

	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED_8
	add  x11, A_ADDR, LDA_B
	add  x12, A_ADDR, LDA_B, lsl #1
	add  x14, A_ADDR, LDA_B, lsl #2
	sub  x13, x14, LDA_B
	add  x15, x11, LDA_B, lsl #2
	add  x16, x12, LDA_B, lsl #2
	add  x17, x15, LDA_B, lsl #1

	ld1  {v8.d}[0],  [A_ADDR]
	ld1  {v8.d}[1],  [x11]
	ld1  {v25.d}[0], [x12]
	ld1  {v25.d}[1], [x13]
	ld1  {v26.d}[0], [x14]
	ld1  {v26.d}[1], [x15]
	ld1  {v27.d}[0], [x16]
	ld1  {v27.d}[1], [x17]

	stp  q8,  q25, [A_REORDER], #32
	stp  q26, q27, [A_REORDER], #32

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
	add  x11, A_ADDR, LDA_B
	add  x12, A_ADDR, LDA_B, lsl #1
	add  x14, A_ADDR, LDA_B, lsl #2
	sub  x13, x14, LDA_B
	add  x15, x11, LDA_B, lsl #2
	add  x16, x12, LDA_B, lsl #2
	add  x17, x15, LDA_B, lsl #1

	ld1  {v0.d}[0], [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ld1  {v1.d}[0], [x12]
	ld1  {v1.d}[1], [x13]
	ld1  {v2.d}[0], [x14]
	ld1  {v2.d}[1], [x15]
	ld1  {v3.d}[0], [x16]
	ld1  {v3.d}[1], [x17]

	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

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

// ── m=1 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_1
	ldr  d0, [A_ADDR]
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

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

// ── m=2 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_2
	add  x11, A_ADDR, LDA_B
	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
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

// ── m=4 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_4
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

// ═══════════════════════════════════════════════════════
// GEMM_BODY_BIAS — unified function body with bias support
//
// Differences from bf16 GEMM_BODY:
//   1. Loads BIAS_ADDR from stack at offset 176 after INIT
//   2. Resets BIAS_ADDR for each M block
//   3. Advances BIAS_ADDR by 32 per N block in epilogue
//   4. Uses LOAD_ZERO_* for accumulator init (no C read)
//   5. B advance per N-block = LDB * 16 (= LDB, lsl #4)
//
// Parameters:
//   \func_name  : symbol name
//   \mrows      : 1 / 2 / 4 / 8
//   \cmpsfx     : 2 / 4 / 8  — cached COMPUTE/LOAD macro suffix
//   \load_z     : LOAD_ZERO macro name
//   \store_c    : STORE_C_*_BIAS_F macro name
//   \ldc_shift  : ldc left-shift (2 = *4 bytes for f32)
//   \c_adv      : C column advance in bytes (32 = 8×f32)
// ═══════════════════════════════════════════════════════

	.macro GEMM_BODY_BIAS func_name, mrows, cmpsfx, load_z, store_c, ldc_shift, c_adv

#ifdef __APPLE__
	.global _\func_name
_\func_name:
#else
	.global \func_name
\func_name:
#endif
	INIT

	// New calling convention: x3 = A_REORDER, x4 = PARAMS, x5 = bias
	mov  A_REORDER, x3              // save A_REORDER → x6
	str  x5, [sp, #-16]!            // push original bias (for M-loop reload)
	mov  BIAS_ADDR, x5              // current bias pointer → x21

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

	// Reset per-M-block state; reload bias pointer
	ldr  BIAS_ADDR, [sp]            // reload original bias from our slot
	mov  A_REORDER_M, A_REORDER
	mov  FIRST_N, #1
	mov  C_N_BASE, C_ROW_BASE
	mov  B_N_BASE, B_ADDR

.L\func_name\()_n_loop:
	asr  N_COUNT, N, #3
	cmp  N_COUNT, #0
	b.eq .L\func_name\()_end

.L\func_name\()_n_iter:
	mov  B_ADDR_1, B_N_BASE
	mov  C_M_BASE, C_N_BASE

	cmp  FIRST_N, #1
	b.eq .L\func_name\()_strided

// ── CACHED PATH ──────────────────────────────────────
	mov  A_ADDR, A_REORDER_M

	\load_z

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

// ── STRIDED PATH ─────────────────────────────────────
.L\func_name\()_strided:
	mov  A_ADDR, A_ORIG_M

	\load_z

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

// ── EPILOGUE ─────────────────────────────────────────
.L\func_name\()_epilogue:
	\store_c
	add  C_N_BASE, C_N_BASE, #\c_adv
	add  B_N_BASE, B_N_BASE, LDB, lsl #4     // B: K*16 bytes per N-block
	add  BIAS_ADDR, BIAS_ADDR, #32           // bias: 8×f32 per N-block

	subs N_COUNT, N_COUNT, #1
	b.gt .L\func_name\()_n_iter

	M_LOOP_END_\mrows

.L\func_name\()_end:
	add  sp, sp, #16                // pop saved bias
	RESTORE
	ret

	.endm   // GEMM_BODY_BIAS

// ═══════════════════════════════════════════════════════
// Function instantiations
// ═══════════════════════════════════════════════════════

// Main kernel (m=8)
GEMM_BODY_BIAS bf16gemm_k_ld_bias_f,  8, 8, LOAD_ZERO_8, STORE_C_ROWMAJOR_BIAS_F, 2, 32

// Tail kernels (m=1,2,4)
GEMM_BODY_BIAS bf16gemm_k_ld1_bias_f, 1, 2, LOAD_ZERO_2, STORE_C_1_BIAS_F,        2, 32
GEMM_BODY_BIAS bf16gemm_k_ld2_bias_f, 2, 2, LOAD_ZERO_2, STORE_C_2_BIAS_F,        2, 32
GEMM_BODY_BIAS bf16gemm_k_ld4_bias_f, 4, 4, LOAD_ZERO_4, STORE_C_4_BIAS_F,        2, 32
