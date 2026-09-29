// i8gemm_hybrid.S -- small-shape "hybrid" i8 microkernel.
//
// Reads A directly from the original row-major layout (no separate A-reorder
// pack pass) using branch-free ldr/ld1/dup, computes an 8-row x 2VL
// tile with the double-buffered smmla schedule, and writes int32 C with the
// contiguous deinterleave store (no scatter). Aimed at tiny shapes where the
// A-pack pass and scatter store dominate.
//
// void i8gemm_k_hybrid(const int8_t *A, const int8_t *B_reo, int32_t *C,
//                      int8_t *unused, const gemm_params_t *p);
//   params: m, k, n, lda, ldb(unused here), ldc.  K is a multiple of 8.
//   Processes M rows in 8-row blocks (m_now = min(M-m0,8)); N in VL/2 tiles.

	.text
	.arch armv8.6-a+sve+i8mm

	.macro HSAVE
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	.endm
	.macro HREST
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16
	.endm

	// Load row-pair rp (rows 2*rp, 2*rp+1) of the current 8-row block into a
	// replicated quad in \zreg. Rows are read directly from A at byte offset
	// (m0 + 2*rp)*lda + kb. Zero-fills rows beyond m_now. x21 = A + m0*lda + kb.
	// x11 = lda, x3 = m_now.
	.macro HLOAD_A zreg, rp
	cmp x3, #(\rp * 2 + 1)
	b.gt .Lhla_\@_\rp\()_both        // both rows present
	cmp x3, #(\rp * 2)
	b.le .Lhla_\@_\rp\()_zero        // no rows
	// only the even row present
	mov x22, #(\rp * 2)
	madd x22, x22, x11, x21
	ldr d0, [x22]
	mov v0.d[1], xzr
	b .Lhla_\@_\rp\()_dup
.Lhla_\@_\rp\()_both:
	mov x22, #(\rp * 2)
	madd x22, x22, x11, x21
	ldr d0, [x22]
	add x22, x22, x11
	ld1 {v0.d}[1], [x22]
.Lhla_\@_\rp\()_dup:
	dup \zreg\().q, z0.q[0]
	b .Lhla_\@_\rp\()_done
.Lhla_\@_\rp\()_zero:
	mov \zreg\().b, #0
.Lhla_\@_\rp\()_done:
	.endm

	.macro HMMLA_ROW areg, c0, c1, c2, c3
	smmla \c0\().s, \areg\().b, z4.b
	smmla \c1\().s, \areg\().b, z5.b
	smmla \c2\().s, \areg\().b, z6.b
	smmla \c3\().s, \areg\().b, z7.b
	.endm

	// Deinterleave-store one row-pair (rows 2*rp,2*rp+1) of the 8xN i32 tile,
	// contiguous st1w (no scatter). x18 = tile base byte addr; x12 = ldc;
	// x3 = m_now; p2 = VL4; p3 masks high quads at VL128; x4/x7 scratch.
	.macro HSTORE_PAIR rp, a0, a1, a2, a3
	cmp x3, #(\rp * 2)
	b.le .Lhs_\@_\rp\()_done
	trn1 z0.d, \a0\().d, \a1\().d
	trn1 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2)
	mul x7, x4, x12
	add x7, x18, x7, lsl #2
	st1w z0.s, p2, [x7]
	add x7, x7, #16
	st1w z1.s, p2, [x7]
	add x7, x7, #16
	mov z2.d, z0.d
	ext z2.b, z2.b, z2.b, #16
	st1w z2.s, p3, [x7]
	add x7, x7, #16
	mov z3.d, z1.d
	ext z3.b, z3.b, z3.b, #16
	st1w z3.s, p3, [x7]
	cmp x3, #(\rp * 2 + 1)
	b.le .Lhs_\@_\rp\()_done
	trn2 z0.d, \a0\().d, \a1\().d
	trn2 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2 + 1)
	mul x7, x4, x12
	add x7, x18, x7, lsl #2
	st1w z0.s, p2, [x7]
	add x7, x7, #16
	st1w z1.s, p2, [x7]
	add x7, x7, #16
	mov z2.d, z0.d
	ext z2.b, z2.b, z2.b, #16
	st1w z2.s, p3, [x7]
	add x7, x7, #16
	mov z3.d, z1.d
	ext z3.b, z3.b, z3.b, #16
	st1w z3.s, p3, [x7]
.Lhs_\@_\rp\()_done:
	.endm

	.global i8gemm_k_hybrid
	.type i8gemm_k_hybrid, %function
i8gemm_k_hybrid:
	HSAVE
	ldr w8,  [x4, #0]         // M
	ldr w9,  [x4, #4]         // K
	ldr w10, [x4, #8]         // N
	ldr w11, [x4, #12]        // lda (bytes per A row)
	ldr w12, [x4, #20]        // ldc
	mov x19, x1               // B_reo base

	ptrue p0.b
	ptrue p2.s, vl4
	cntw x4
	sub x4, x4, #4
	whilelt p3.s, xzr, x4

	cntb x13
	lsr x13, x13, #1          // n_tile = VL/2 (cols)
	mul x14, x9, x13          // B N-tile stride in bytes (K * n_tile)

	mov x15, #0               // m0
.Lh_m:
	cmp x15, x8
	b.ge .Lh_done
	sub x3, x8, x15
	cmp x3, #8
	mov x20, #8
	csel x3, x3, x20, le      // m_now = min(M-m0, 8)

	mov x16, #0               // n0
	mov x17, x19              // current B tile
.Lh_n:
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	mov z20.s, #0
	mov z21.s, #0
	mov z22.s, #0
	mov z23.s, #0
	mov z24.s, #0
	mov z25.s, #0
	mov z26.s, #0
	mov z27.s, #0
	mov z28.s, #0
	mov z29.s, #0
	mov z30.s, #0
	mov z31.s, #0

	mov x6, x17               // B K pointer
	// x21 = A + m0*lda  (advanced by kb inside the loop)
	mul x21, x15, x11
	add x21, x0, x21
	mov x4, #0                // kb
.Lh_k:
	cmp x4, x9
	b.ge .Lh_store
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	prfm pldl1keep, [x6, #512]
	HLOAD_A z0, 0
	HMMLA_ROW z0, z16, z17, z18, z19
	HLOAD_A z1, 1
	HMMLA_ROW z1, z20, z21, z22, z23
	HLOAD_A z2, 2
	HMMLA_ROW z2, z24, z25, z26, z27
	HLOAD_A z3, 3
	HMMLA_ROW z3, z28, z29, z30, z31
	add x4, x4, #8
	add x21, x21, #8
	addvl x6, x6, #4
	b .Lh_k

.Lh_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2
	HSTORE_PAIR 0, z16, z17, z18, z19
	HSTORE_PAIR 1, z20, z21, z22, z23
	HSTORE_PAIR 2, z24, z25, z26, z27
	HSTORE_PAIR 3, z28, z29, z30, z31

	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .Lh_n

	add x15, x15, #8
	b .Lh_m
.Lh_done:
	HREST
	ret
	.size i8gemm_k_hybrid, .-i8gemm_k_hybrid

// ── Narrow-M kernel for M in {1,2} ───────────────────────────────────────
// Same ABI, same B_reo layout as i8gemm_k_hybrid. For M<=2 the 8-row tile
// wastes (8-M)/8 of every smmla; here we process a single row-pair (rows 0,1),
// issuing only 4 smmla per K8 step instead of 16. int32 output, contiguous
// deinterleave store. Reuses HLOAD_A/HMMLA_ROW/HSTORE_PAIR (m_now-aware).
	.global i8gemm_k_narrow
	.type i8gemm_k_narrow, %function
i8gemm_k_narrow:
	HSAVE
	ldr w8,  [x4, #0]         // M (1 or 2)
	ldr w9,  [x4, #4]         // K
	ldr w10, [x4, #8]         // N
	ldr w11, [x4, #12]        // lda
	ldr w12, [x4, #20]        // ldc
	mov x19, x1               // B_reo base

	ptrue p0.b
	ptrue p2.s, vl4
	cntw x4
	sub x4, x4, #4
	whilelt p3.s, xzr, x4

	cntb x13
	lsr x13, x13, #1          // n_tile = VL/2
	mul x14, x9, x13          // B N-tile stride

	mov x3, x8                // m_now = M (<=2); drives HLOAD_A / HSTORE_PAIR
	mov x15, #0               // m0 = 0 (single row block)
	mov x16, #0               // n0
	mov x17, x19              // current B tile
.Lnw_n:
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	mov x6, x17               // B K pointer
	mov x21, x0               // A base (m0=0), advanced by kb
	mov x4, #0                // kb
.Lnw_k:
	cmp x4, x9
	b.ge .Lnw_store
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	prfm pldl1keep, [x6, #512]
	HLOAD_A z0, 0
	HMMLA_ROW z0, z16, z17, z18, z19
	add x4, x4, #8
	add x21, x21, #8
	addvl x6, x6, #4
	b .Lnw_k
.Lnw_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2
	HSTORE_PAIR 0, z16, z17, z18, z19
	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .Lnw_n
	HREST
	ret
	.size i8gemm_k_narrow, .-i8gemm_k_narrow

// ── Narrow-M kernel for M in {3,4} ───────────────────────────────────────
// Two row-pairs (rows 0..3), 8 smmla per K8 step (vs the 8-row tile's 16).
// Branch-free (M=3 zero-fills row 3 via HLOAD_A/HSTORE_PAIR self-skip). For
// M=3/4 this beats both the 8-row hybrid (pad-to-8, +22-24% measured on V1)
// and the 1-pair kernel. M>=5 stays on the hybrid (its wider B-amortisation
// and straight-line pipeline win past the crossover).
	.global i8gemm_k_narrow2
	.type i8gemm_k_narrow2, %function
i8gemm_k_narrow2:
	HSAVE
	ldr w8,  [x4, #0]         // M (3 or 4)
	ldr w9,  [x4, #4]         // K
	ldr w10, [x4, #8]         // N
	ldr w11, [x4, #12]        // lda
	ldr w12, [x4, #20]        // ldc
	mov x19, x1               // B_reo base

	ptrue p0.b
	ptrue p2.s, vl4
	cntw x4
	sub x4, x4, #4
	whilelt p3.s, xzr, x4

	cntb x13
	lsr x13, x13, #1          // n_tile = VL/2
	mul x14, x9, x13          // B N-tile stride

	mov x3, x8                // m_now = M (<=4)
	mov x15, #0               // m0 = 0
	mov x16, #0               // n0
	mov x17, x19              // current B tile
.Ln2_n:
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	mov z20.s, #0
	mov z21.s, #0
	mov z22.s, #0
	mov z23.s, #0
	mov x6, x17               // B K pointer
	mov x21, x0               // A base (m0=0), advanced by kb
	mov x4, #0                // kb
.Ln2_k:
	cmp x4, x9
	b.ge .Ln2_store
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	prfm pldl1keep, [x6, #512]
	HLOAD_A z0, 0
	HMMLA_ROW z0, z16, z17, z18, z19
	HLOAD_A z1, 1
	HMMLA_ROW z1, z20, z21, z22, z23
	add x4, x4, #8
	add x21, x21, #8
	addvl x6, x6, #4
	b .Ln2_k
.Ln2_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2
	HSTORE_PAIR 0, z16, z17, z18, z19
	HSTORE_PAIR 1, z20, z21, z22, z23
	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .Ln2_n
	HREST
	ret
	.size i8gemm_k_narrow2, .-i8gemm_k_narrow2

	.section .note.GNU-stack,"",@progbits
