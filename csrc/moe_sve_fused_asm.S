// fused_cpp-owned SVE BF16 MoE kernels.
//
// This file intentionally does not export the upstream bf16gemm_k_* symbols.
// The ABI matches the existing low-level GEMM entry points:
//   x0 packed A (reorder-m8)
//   x1 packed B (SVE N-tile layout)
//   x2 C
//   x3 unused A_reorder slot
//   x4 gemm_params_t*
//
// Each entry computes a single M tile: m8/m4/m2/m1. The C++ dispatcher owns
// the outer M loop and tail policy.

	.text
	.align 4

	.macro SAVE_REGS
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

	.macro RESTORE_REGS
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

	.macro ADR_CONST reg, label
#ifdef __APPLE__
	adrp \reg, \label@PAGE
	add  \reg, \reg, \label@PAGEOFF
#else
	adrp \reg, \label
	add  \reg, \reg, :lo12:\label
#endif
	.endm

	.macro ZERO_ACC_M8
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
	.endm

	.macro ZERO_ACC_M4
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	mov z20.s, #0
	mov z21.s, #0
	mov z22.s, #0
	mov z23.s, #0
	.endm

	.macro ZERO_ACC_M2
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	.endm

	.macro LOAD_AB_PACKED a0, a1, a2, a3, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [x17]
	ld1h {\b1\().h}, p0/z, [x17, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [x17, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {\a0\().h}, p0/z, [x18]
	ld1rqh {\a1\().h}, p0/z, [x18, #16]
	ld1rqh {\a2\().h}, p0/z, [x18, #32]
	ld1rqh {\a3\().h}, p0/z, [x18, #48]
	add x18, x18, #64
	.endm

	.macro LOAD_AB_PACKED_M4 a0, a1, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [x17]
	ld1h {\b1\().h}, p0/z, [x17, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [x17, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {\a0\().h}, p0/z, [x18]
	ld1rqh {\a1\().h}, p0/z, [x18, #16]
	add x18, x18, #64
	.endm

	.macro LOAD_AB_PACKED_M2 a0, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [x17]
	ld1h {\b1\().h}, p0/z, [x17, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [x17, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {\a0\().h}, p0/z, [x18]
	add x18, x18, #64
	.endm

	.macro BFMMLA_BLOCK_AB a0, a1, a2, a3, b0, b1, b2, b3
	bfmmla z16.s, \a0\().h, \b0\().h
	bfmmla z20.s, \a1\().h, \b0\().h
	bfmmla z24.s, \a2\().h, \b0\().h
	bfmmla z28.s, \a3\().h, \b0\().h
	bfmmla z17.s, \a0\().h, \b1\().h
	bfmmla z21.s, \a1\().h, \b1\().h
	bfmmla z25.s, \a2\().h, \b1\().h
	bfmmla z29.s, \a3\().h, \b1\().h
	bfmmla z18.s, \a0\().h, \b2\().h
	bfmmla z22.s, \a1\().h, \b2\().h
	bfmmla z26.s, \a2\().h, \b2\().h
	bfmmla z30.s, \a3\().h, \b2\().h
	bfmmla z19.s, \a0\().h, \b3\().h
	bfmmla z23.s, \a1\().h, \b3\().h
	bfmmla z27.s, \a2\().h, \b3\().h
	bfmmla z31.s, \a3\().h, \b3\().h
	.endm

	.macro BFMMLA_M4_AB a0, a1, b0, b1, b2, b3
	bfmmla z16.s, \a0\().h, \b0\().h
	bfmmla z20.s, \a1\().h, \b0\().h
	bfmmla z17.s, \a0\().h, \b1\().h
	bfmmla z21.s, \a1\().h, \b1\().h
	bfmmla z18.s, \a0\().h, \b2\().h
	bfmmla z22.s, \a1\().h, \b2\().h
	bfmmla z19.s, \a0\().h, \b3\().h
	bfmmla z23.s, \a1\().h, \b3\().h
	.endm

	.macro BFMMLA_M2_AB a0, b0, b1, b2, b3
	bfmmla z16.s, \a0\().h, \b0\().h
	bfmmla z17.s, \a0\().h, \b1\().h
	bfmmla z18.s, \a0\().h, \b2\().h
	bfmmla z19.s, \a0\().h, \b3\().h
	.endm

	.macro ZERO_ACC_M12
	mov z8.s, #0
	mov z9.s, #0
	mov z10.s, #0
	mov z11.s, #0
	mov z12.s, #0
	mov z13.s, #0
	mov z14.s, #0
	mov z15.s, #0
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
	.endm

	.macro BFMMLA_M12_2ROWS a0, a1, c00, c01, c02, c03, c10, c11, c12, c13
	bfmmla \c00\().s, \a0\().h, z4.h
	bfmmla \c10\().s, \a1\().h, z4.h
	bfmmla \c01\().s, \a0\().h, z5.h
	bfmmla \c11\().s, \a1\().h, z5.h
	bfmmla \c02\().s, \a0\().h, z6.h
	bfmmla \c12\().s, \a1\().h, z6.h
	bfmmla \c03\().s, \a0\().h, z7.h
	bfmmla \c13\().s, \a1\().h, z7.h
	.endm

	.macro BFMMLA_M12_2A
	bfmmla z24.s, z0.h, z4.h
	bfmmla z28.s, z1.h, z4.h
	bfmmla z25.s, z0.h, z5.h
	bfmmla z29.s, z1.h, z5.h
	bfmmla z26.s, z0.h, z6.h
	bfmmla z30.s, z1.h, z6.h
	bfmmla z27.s, z0.h, z7.h
	bfmmla z31.s, z1.h, z7.h
	.endm

	.macro M12_K4_BODY
	ld1h {z4.h}, p0/z, [x17]
	ld1h {z5.h}, p0/z, [x17, #1, mul vl]
	ld1h {z6.h}, p0/z, [x17, #2, mul vl]
	ld1h {z7.h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {z0.h}, p0/z, [x18]
	ld1rqh {z1.h}, p0/z, [x18, #16]
	ld1rqh {z2.h}, p0/z, [x18, #32]
	ld1rqh {z3.h}, p0/z, [x18, #48]
	BFMMLA_M12_2ROWS z0, z1, z8, z9, z10, z11, z12, z13, z14, z15
	ld1rqh {z0.h}, p0/z, [x18, #64]
	ld1rqh {z1.h}, p0/z, [x18, #80]
	BFMMLA_M12_2ROWS z2, z3, z16, z17, z18, z19, z20, z21, z22, z23
	BFMMLA_M12_2A
	add x18, x18, #96
	.endm

	.macro BUILD_C_OFFSETS
	index z14.s, #0, #1
	mov z15.d, z14.d
	and z15.s, z15.s, #3
	mov z9.d, z14.d
	lsr z9.s, z9.s, #2
	lsl z9.s, z9.s, #5
	mov z13.d, z15.d
	and z13.s, z13.s, #1
	lsl z13.s, z13.s, #2
	add z9.s, z9.s, z13.s
	mov z12.d, z15.d
	lsr z12.s, z12.s, #1
	and z12.s, z12.s, #1
	cmpeq p4.s, p1/z, z12.s, #0
	dup z13.s, w25
	mul z12.s, p1/m, z12.s, z13.s
	add z9.s, z9.s, z12.s
	.endm

	.macro STORE_PAIR_F32 base, pred, a0, a1, a2, a3
	add x3, \base, #8
	add x4, \base, #16
	add x5, \base, #24
	st1w {\a0\().s}, \pred, [\base, z9.s, sxtw]
	st1w {\a1\().s}, \pred, [x3, z9.s, sxtw]
	st1w {\a2\().s}, \pred, [x4, z9.s, sxtw]
	st1w {\a3\().s}, \pred, [x5, z9.s, sxtw]
	.endm

	.macro STORE_W2_PAIR base, row, a0, a1, a2, a3
	cmp w19, #\row
	b.le 940\@f
	cmp w19, #(\row + 1)
	b.eq 941\@f
	STORE_PAIR_F32 \base, p1, \a0, \a1, \a2, \a3
	b 940\@f
941\@:
	STORE_PAIR_F32 \base, p4, \a0, \a1, \a2, \a3
940\@:
	.endm

	.macro BUILD_PACKC_OFFSETS
	index z0.s, #0, #1
	mov z1.d, z0.d
	lsr z1.s, z1.s, #2
	lsl z1.s, z1.s, #6       // segment * 64 bytes
	mov z2.d, z0.d
	and z2.s, z2.s, #3
	mov z3.d, z2.d
	and z3.s, z3.s, #1
	lsl z3.s, z3.s, #1       // col_bit * sizeof(bf16)
	add z5.s, z1.s, z3.s
	mov z4.d, z2.d
	lsr z4.s, z4.s, #1
	and z4.s, z4.s, #1       // row_bit inside the row pair
	cmpeq p4.s, p1/z, z4.s, #0
	lsl z4.s, z4.s, #3       // row_bit * 4 bf16 values
	add z5.s, z5.s, z4.s
	.endm

	.macro BUILD_PACKC_OFFSETS_M12
	index z0.s, #0, #1
	mov z1.d, z0.d
	lsr z1.s, z1.s, #2
	mov z5.d, z1.d
	lsl z1.s, z1.s, #6       // segment * 64 bytes
	lsl z5.s, z5.s, #5       // segment * 32 bytes
	add z5.s, z5.s, z1.s     // segment * 96 bytes
	mov z2.d, z0.d
	and z2.s, z2.s, #3
	mov z3.d, z2.d
	and z3.s, z3.s, #1
	lsl z3.s, z3.s, #1       // col_bit * sizeof(bf16)
	add z5.s, z5.s, z3.s
	mov z4.d, z2.d
	lsr z4.s, z4.s, #1
	and z4.s, z4.s, #1       // row_bit inside the row pair
	lsl z4.s, z4.s, #3       // row_bit * 4 bf16 values
	add z5.s, z5.s, z4.s
	.endm

	.macro SILU_CONSTS_LOAD
	ADR_CONST x3, .Lmoe_sve_silu_consts
	ld1rw {z6.s}, p1/z, [x3]      // 1.0
	add x3, x3, #4
	ld1rw {z7.s}, p1/z, [x3]      // 0.5
	add x3, x3, #4
	ld1rw {z8.s}, p1/z, [x3]      // invln2
	add x3, x3, #4
	ld1rw {z9.s}, p1/z, [x3]      // ln2
	add x3, x3, #4
	ld1rw {z10.s}, p1/z, [x3]     // c3
	add x3, x3, #4
	ld1rw {z11.s}, p1/z, [x3]     // c4
	add x3, x3, #4
	ld1rw {z12.s}, p1/z, [x3]     // c5
	add x3, x3, #4
	ld1rw {z13.s}, p1/z, [x3]     // c6
	add x3, x3, #4
	ld1rw {z14.s}, p1/z, [x3]     // clamp_hi
	add x3, x3, #4
	ld1rw {z15.s}, p1/z, [x3]     // clamp_lo
	.endm

	.macro SILU_TO_BF16 gate, up, deg
	fneg z0.s, p1/m, \gate\().s
	fmin z0.s, p1/m, z0.s, z14.s
	fmax z0.s, p1/m, z0.s, z15.s
	mov z1.d, z0.d
	fmul z1.s, p1/m, z1.s, z8.s
	frintn z1.s, p1/m, z1.s
	fcvtzs z2.s, p1/m, z1.s
	fmls z0.s, p1/m, z1.s, z9.s
	.if \deg == 4
	mov z1.d, z11.d
	.elseif \deg == 5
	mov z3.d, z12.d
	mov z1.d, z11.d
	fmla z1.s, p1/m, z3.s, z0.s
	.else
	mov z1.d, z13.d
	mov z3.d, z12.d
	fmla z3.s, p1/m, z1.s, z0.s
	mov z1.d, z11.d
	fmla z1.s, p1/m, z3.s, z0.s
	.endif
	mov z3.d, z10.d
	fmla z3.s, p1/m, z1.s, z0.s
	mov z1.d, z7.d
	fmla z1.s, p1/m, z3.s, z0.s
	mov z3.d, z6.d
	fmla z3.s, p1/m, z1.s, z0.s
	mov z1.d, z6.d
	fmla z1.s, p1/m, z3.s, z0.s
	mov z4.s, #127
	add z2.s, z2.s, z4.s
	lsl z2.s, z2.s, #23
	fmul z1.s, p1/m, z1.s, z2.s
	fadd z1.s, p1/m, z1.s, z6.s
	mov z0.d, \gate\().d
	fmul z0.s, p1/m, z0.s, \up\().s
	fdiv z0.s, p1/m, z0.s, z1.s
	bfcvt z0.h, p1/m, z0.s
	.endm

	.macro LOAD_SILU_CONST zd, byteoff
	ADR_CONST x3, .Lmoe_sve_silu_consts
	.if \byteoff != 0
	add x3, x3, #\byteoff
	.endif
	ld1rw {\zd\().s}, p1/z, [x3]
	.endm

	.macro SILU_TO_BF16_LOW_TMP gate, up, deg
	fneg z0.s, p1/m, \gate\().s
	LOAD_SILU_CONST z4, 32
	fmin z0.s, p1/m, z0.s, z4.s
	LOAD_SILU_CONST z4, 36
	fmax z0.s, p1/m, z0.s, z4.s
	LOAD_SILU_CONST z4, 8
	mov z1.d, z0.d
	fmul z1.s, p1/m, z1.s, z4.s
	frintn z1.s, p1/m, z1.s
	fcvtzs z2.s, p1/m, z1.s
	LOAD_SILU_CONST z4, 12
	fmls z0.s, p1/m, z1.s, z4.s
	.if \deg == 4
	LOAD_SILU_CONST z3, 20
	.elseif \deg == 5
	LOAD_SILU_CONST z3, 24
	LOAD_SILU_CONST z4, 20
	fmla z4.s, p1/m, z3.s, z0.s
	mov z3.d, z4.d
	.else
	LOAD_SILU_CONST z3, 28
	LOAD_SILU_CONST z4, 24
	fmla z4.s, p1/m, z3.s, z0.s
	LOAD_SILU_CONST z3, 20
	fmla z3.s, p1/m, z4.s, z0.s
	.endif
	LOAD_SILU_CONST z4, 16
	fmla z4.s, p1/m, z3.s, z0.s
	LOAD_SILU_CONST z3, 4
	fmla z3.s, p1/m, z4.s, z0.s
	LOAD_SILU_CONST z4, 0
	fmla z4.s, p1/m, z3.s, z0.s
	LOAD_SILU_CONST z3, 0
	fmla z3.s, p1/m, z4.s, z0.s
	mov z4.s, #127
	add z2.s, z2.s, z4.s
	lsl z2.s, z2.s, #23
	fmul z3.s, p1/m, z3.s, z2.s
	LOAD_SILU_CONST z4, 0
	fadd z3.s, p1/m, z3.s, z4.s
	mov z0.d, \gate\().d
	fmul z0.s, p1/m, z0.s, \up\().s
	fdiv z0.s, p1/m, z0.s, z3.s
	bfcvt z0.h, p1/m, z0.s
	.endm

	.macro STORE_PACKC_PAIR base, pred, g0, g1, u0, u1, deg
	SILU_TO_BF16 \g0, \u0, \deg
	st1h {z0.s}, \pred, [\base, z5.s, sxtw]
	SILU_TO_BF16 \g1, \u1, \deg
	add x3, \base, #4
	st1h {z0.s}, \pred, [x3, z5.s, sxtw]
	.endm

	.macro STORE_PACKC_PAIR_GUARDED base, row, g0, g1, u0, u1, deg
	cmp w19, #\row
	b.le 950\@f
	cmp w19, #(\row + 1)
	b.eq 951\@f
	STORE_PACKC_PAIR \base, p1, \g0, \g1, \u0, \u1, \deg
	b 950\@f
951\@:
	STORE_PACKC_PAIR \base, p4, \g0, \g1, \u0, \u1, \deg
950\@:
	.endm

	.macro STORE_PACKC_M8 deg
	BUILD_PACKC_OFFSETS
	SILU_CONSTS_LOAD
	mov x16, x12
	STORE_PACKC_PAIR_GUARDED x16, 0, z16, z17, z18, z19, \deg
	add x16, x12, #16
	STORE_PACKC_PAIR_GUARDED x16, 2, z20, z21, z22, z23, \deg
	add x16, x12, #32
	STORE_PACKC_PAIR_GUARDED x16, 4, z24, z25, z26, z27, \deg
	add x16, x12, #48
	STORE_PACKC_PAIR_GUARDED x16, 6, z28, z29, z30, z31, \deg
	.endm

	.macro STORE_PACKC_M4 deg
	BUILD_PACKC_OFFSETS
	SILU_CONSTS_LOAD
	mov x16, x12
	STORE_PACKC_PAIR_GUARDED x16, 0, z16, z17, z18, z19, \deg
	add x16, x12, #16
	STORE_PACKC_PAIR_GUARDED x16, 2, z20, z21, z22, z23, \deg
	.endm

	.macro STORE_PACKC_M2 deg
	BUILD_PACKC_OFFSETS
	SILU_CONSTS_LOAD
	mov x16, x12
	STORE_PACKC_PAIR_GUARDED x16, 0, z16, z17, z18, z19, \deg
	.endm

	.macro STORE_PACKC_PAIR_M12 base, g0, g1, u0, u1, deg
	SILU_TO_BF16_LOW_TMP \g0, \u0, \deg
	st1h {z0.s}, p1, [\base, z5.s, sxtw]
	SILU_TO_BF16_LOW_TMP \g1, \u1, \deg
	add x3, \base, #4
	st1h {z0.s}, p1, [x3, z5.s, sxtw]
	.endm

	.macro STORE_PACKC_M12 deg
	BUILD_PACKC_OFFSETS_M12
	mov x16, x12
	STORE_PACKC_PAIR_M12 x16, z8, z9, z10, z11, \deg
	add x16, x12, #16
	STORE_PACKC_PAIR_M12 x16, z12, z13, z14, z15, \deg
	add x16, x12, #32
	STORE_PACKC_PAIR_M12 x16, z16, z17, z18, z19, \deg
	add x16, x12, #48
	STORE_PACKC_PAIR_M12 x16, z20, z21, z22, z23, \deg
	add x16, x12, #64
	STORE_PACKC_PAIR_M12 x16, z24, z25, z26, z27, \deg
	add x16, x12, #80
	STORE_PACKC_PAIR_M12 x16, z28, z29, z30, z31, \deg
	.endm

	.macro STORE_W2_M8
	BUILD_C_OFFSETS
	mov x16, x12
	STORE_W2_PAIR x16, 0, z16, z17, z18, z19
	add x16, x12, x25, lsl #1
	STORE_W2_PAIR x16, 2, z20, z21, z22, z23
	add x16, x12, x25, lsl #2
	STORE_W2_PAIR x16, 4, z24, z25, z26, z27
	add x16, x16, x25, lsl #1
	STORE_W2_PAIR x16, 6, z28, z29, z30, z31
	.endm

	.macro STORE_W2_M4
	BUILD_C_OFFSETS
	mov x16, x12
	STORE_W2_PAIR x16, 0, z16, z17, z18, z19
	add x16, x12, x25, lsl #1
	STORE_W2_PAIR x16, 2, z20, z21, z22, z23
	.endm

	.macro STORE_W2_M2
	BUILD_C_OFFSETS
	mov x16, x12
	STORE_W2_PAIR x16, 0, z16, z17, z18, z19
	.endm

	.macro BUILD_C_OFFSETS_M12
	index z0.s, #0, #1
	mov z1.d, z0.d
	and z1.s, z1.s, #3
	mov z2.d, z0.d
	lsr z2.s, z2.s, #2
	lsl z2.s, z2.s, #5
	mov z3.d, z1.d
	and z3.s, z3.s, #1
	lsl z3.s, z3.s, #2
	add z0.s, z2.s, z3.s
	mov z2.d, z1.d
	lsr z2.s, z2.s, #1
	and z2.s, z2.s, #1
	dup z3.s, w25
	mul z2.s, p1/m, z2.s, z3.s
	add z0.s, z0.s, z2.s
	.endm

	.macro STORE_PAIR_W2_M12 base, a0, a1, a2, a3
	add x3, \base, #8
	add x4, \base, #16
	add x5, \base, #24
	st1w {\a0\().s}, p1, [\base, z0.s, sxtw]
	st1w {\a1\().s}, p1, [x3, z0.s, sxtw]
	st1w {\a2\().s}, p1, [x4, z0.s, sxtw]
	st1w {\a3\().s}, p1, [x5, z0.s, sxtw]
	.endm

	.macro STORE_W2_M12
	BUILD_C_OFFSETS_M12
	mov x16, x12
	STORE_PAIR_W2_M12 x16, z8, z9, z10, z11
	add x16, x12, x25, lsl #1
	STORE_PAIR_W2_M12 x16, z12, z13, z14, z15
	add x16, x12, x25, lsl #2
	STORE_PAIR_W2_M12 x16, z16, z17, z18, z19
	add x16, x16, x25, lsl #1
	STORE_PAIR_W2_M12 x16, z20, z21, z22, z23
	add x16, x12, x25, lsl #3
	STORE_PAIR_W2_M12 x16, z24, z25, z26, z27
	add x16, x16, x25, lsl #1
	STORE_PAIR_W2_M12 x16, z28, z29, z30, z31
	.endm

	.p2align 4
moe_sve_kernel_m12_rows:
	SAVE_REGS
	ldr w22, [x4]             // total M rows, multiple of 12
	ldr w20, [x4, #4]         // K
	ldr w21, [x4, #8]         // N columns in this slice
	ldr w23, [x4, #20]        // ldc
	mov x28, x0               // current packed A block
	mov x29, x2               // current C block
	cntb x27
	lsr x9, x27, #1           // SVE N tile in bf16 columns
	mul x13, x20, x27         // packed-B bytes per SVE N tile
	lsl x24, x27, #2
	add x24, x24, x27, lsl #1 // M12 packC advance per N tile: VL * 6 bytes
	lsl x26, x20, #4
	add x26, x26, x20, lsl #3 // A advance per M12 block: K * 24 bytes
	lsl x19, x23, #4
	add x19, x19, x23, lsl #3 // C advance per M12 block: ldc * 24 bytes
	ptrue p0.b
	ptrue p1.s

.L_m12_rows_loop:
	cmp w22, #12
	b.lt .L_m12_rows_done
	mov x11, x1
	mov x12, x29
	mov w10, w21

.L_m12_rows_n_loop:
	cmp w10, #0
	b.le .L_m12_rows_next_m
	ZERO_ACC_M12
	mov x17, x11
	mov x18, x28
	mov w15, w20

.L_m12_rows_k_loop:
	M12_K4_BODY
	M12_K4_BODY
	subs w15, w15, #8
	b.gt .L_m12_rows_k_loop

	cmp w6, #4
	b.eq .L_m12_rows_store_p4
	cmp w6, #5
	b.eq .L_m12_rows_store_p5
	STORE_PACKC_M12 6
	b .L_m12_rows_advance_n
.L_m12_rows_store_p4:
	STORE_PACKC_M12 4
	b .L_m12_rows_advance_n
.L_m12_rows_store_p5:
	STORE_PACKC_M12 5

.L_m12_rows_advance_n:
	add x12, x12, x24
	add x11, x11, x13
	sub w10, w10, w9
	b .L_m12_rows_n_loop

.L_m12_rows_next_m:
	add x28, x28, x26
	add x29, x29, x19
	sub w22, w22, #12
	b .L_m12_rows_loop

.L_m12_rows_done:
	RESTORE_REGS
	ret

	.p2align 4
moe_sve_kernel_m12:
	SAVE_REGS
	ldr w20, [x4, #4]         // K
	ldr w21, [x4, #8]         // N columns in this slice
	ldr w23, [x4, #20]        // ldc
	mov x28, x0               // packed A
	cntb x27
	lsr x9, x27, #1           // SVE N tile in bf16 columns
	lsl x25, x23, #2          // f32 ldc bytes
	mul x13, x20, x27         // packed-B bytes per SVE N tile
	lsl x14, x27, #1          // row-major f32 C advance per N tile
	lsl x24, x27, #2
	add x24, x24, x27, lsl #1 // M12 packC advance per N tile: VL * 6 bytes
	ptrue p0.b
	ptrue p1.s
	mov x11, x1
	mov x12, x2
	mov w10, w21

.L_m12_n_loop:
	cmp w10, #0
	b.le .L_m12_done
	ZERO_ACC_M12
	mov x17, x11
	mov x18, x28
	mov w15, w20

.L_m12_k_loop:
	M12_K4_BODY
	M12_K4_BODY
	subs w15, w15, #8
	b.gt .L_m12_k_loop

	cmp w6, #0
	b.eq .L_m12_store_w2
	cmp w6, #4
	b.eq .L_m12_store_p4
	cmp w6, #5
	b.eq .L_m12_store_p5
	STORE_PACKC_M12 6
	b .L_m12_advance_packc
.L_m12_store_p4:
	STORE_PACKC_M12 4
	b .L_m12_advance_packc
.L_m12_store_p5:
	STORE_PACKC_M12 5
	b .L_m12_advance_packc
.L_m12_store_w2:
	STORE_W2_M12
	add x12, x12, x14
	b .L_m12_advance_common
.L_m12_advance_packc:
	add x12, x12, x24
.L_m12_advance_common:
	add x11, x11, x13
	sub w10, w10, w9
	b .L_m12_n_loop

.L_m12_done:
	RESTORE_REGS
	ret

	.p2align 4
moe_sve_kernel_m8:
	SAVE_REGS
	ldp w19, w20, [x4]        // M rows in this tile, K
	ldr w21, [x4, #8]         // N columns in this slice
	ldr w23, [x4, #20]        // ldc
	mov x28, x0               // packed A
	cntb x27
	lsr x9, x27, #1           // SVE N tile in bf16 columns
	lsl x25, x23, #2          // f32 ldc bytes
	mul x13, x20, x27         // packed-B bytes per SVE N tile
	lsl x14, x27, #1          // row-major f32 C advance per N tile
	lsl x24, x27, #2          // packC advance per N tile
	ptrue p0.b
	ptrue p1.s
	mov x11, x1
	mov x12, x2
	mov w10, w21

.L_m8_n_loop:
	cmp w10, #0
	b.le .L_m8_done
	ZERO_ACC_M8
	mov x17, x11
	mov x18, x28
	mov w15, w20

	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m8_tail_cur

.L_m8_k_loop:
	LOAD_AB_PACKED z8, z9, z10, z11, z12, z13, z14, z15
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m8_tail_next
	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_m8_k_loop

.L_m8_tail_cur:
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	b .L_m8_store

.L_m8_tail_next:
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15

.L_m8_store:
	cmp w6, #0
	b.eq .L_m8_store_w2
	cmp w6, #4
	b.eq .L_m8_store_p4
	cmp w6, #5
	b.eq .L_m8_store_p5
	STORE_PACKC_M8 6
	b .L_m8_advance_packc
.L_m8_store_p4:
	STORE_PACKC_M8 4
	b .L_m8_advance_packc
.L_m8_store_p5:
	STORE_PACKC_M8 5
	b .L_m8_advance_packc
.L_m8_store_w2:
	STORE_W2_M8
	add x12, x12, x14
	b .L_m8_advance_common
.L_m8_advance_packc:
	add x12, x12, x24
.L_m8_advance_common:
	add x11, x11, x13
	sub w10, w10, w9
	b .L_m8_n_loop

.L_m8_done:
	RESTORE_REGS
	ret

	.p2align 4
moe_sve_kernel_m4:
	SAVE_REGS
	ldp w19, w20, [x4]
	ldr w21, [x4, #8]
	ldr w23, [x4, #20]
	mov x28, x0
	cntb x27
	lsr x9, x27, #1
	lsl x25, x23, #2
	mul x13, x20, x27
	lsl x14, x27, #1
	lsl x24, x27, #2
	ptrue p0.b
	ptrue p1.s
	mov x11, x1
	mov x12, x2
	mov w10, w21

.L_m4_n_loop:
	cmp w10, #0
	b.le .L_m4_done
	ZERO_ACC_M4
	mov x17, x11
	mov x18, x28
	mov w15, w20

	LOAD_AB_PACKED_M4 z0, z1, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m4_tail_cur

.L_m4_k_loop:
	LOAD_AB_PACKED_M4 z8, z9, z12, z13, z14, z15
	BFMMLA_M4_AB z0, z1, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m4_tail_next
	LOAD_AB_PACKED_M4 z0, z1, z4, z5, z6, z7
	BFMMLA_M4_AB z8, z9, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_m4_k_loop

.L_m4_tail_cur:
	BFMMLA_M4_AB z0, z1, z4, z5, z6, z7
	b .L_m4_store

.L_m4_tail_next:
	BFMMLA_M4_AB z8, z9, z12, z13, z14, z15

.L_m4_store:
	cmp w6, #0
	b.eq .L_m4_store_w2
	cmp w6, #4
	b.eq .L_m4_store_p4
	cmp w6, #5
	b.eq .L_m4_store_p5
	STORE_PACKC_M4 6
	b .L_m4_advance_packc
.L_m4_store_p4:
	STORE_PACKC_M4 4
	b .L_m4_advance_packc
.L_m4_store_p5:
	STORE_PACKC_M4 5
	b .L_m4_advance_packc
.L_m4_store_w2:
	STORE_W2_M4
	add x12, x12, x14
	b .L_m4_advance_common
.L_m4_advance_packc:
	add x12, x12, x24
.L_m4_advance_common:
	add x11, x11, x13
	sub w10, w10, w9
	b .L_m4_n_loop

.L_m4_done:
	RESTORE_REGS
	ret

	.p2align 4
moe_sve_kernel_m2:
	SAVE_REGS
	ldp w19, w20, [x4]
	ldr w21, [x4, #8]
	ldr w23, [x4, #20]
	mov x28, x0
	cntb x27
	lsr x9, x27, #1
	lsl x25, x23, #2
	mul x13, x20, x27
	lsl x14, x27, #1
	lsl x24, x27, #2
	ptrue p0.b
	ptrue p1.s
	mov x11, x1
	mov x12, x2
	mov w10, w21

.L_m2_n_loop:
	cmp w10, #0
	b.le .L_m2_done
	ZERO_ACC_M2
	mov x17, x11
	mov x18, x28
	mov w15, w20

	LOAD_AB_PACKED_M2 z0, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m2_tail_cur

.L_m2_k_loop:
	LOAD_AB_PACKED_M2 z8, z12, z13, z14, z15
	BFMMLA_M2_AB z0, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m2_tail_next
	LOAD_AB_PACKED_M2 z0, z4, z5, z6, z7
	BFMMLA_M2_AB z8, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_m2_k_loop

.L_m2_tail_cur:
	BFMMLA_M2_AB z0, z4, z5, z6, z7
	b .L_m2_store

.L_m2_tail_next:
	BFMMLA_M2_AB z8, z12, z13, z14, z15

.L_m2_store:
	cmp w6, #0
	b.eq .L_m2_store_w2
	cmp w6, #4
	b.eq .L_m2_store_p4
	cmp w6, #5
	b.eq .L_m2_store_p5
	STORE_PACKC_M2 6
	b .L_m2_advance_packc
.L_m2_store_p4:
	STORE_PACKC_M2 4
	b .L_m2_advance_packc
.L_m2_store_p5:
	STORE_PACKC_M2 5
	b .L_m2_advance_packc
.L_m2_store_w2:
	STORE_W2_M2
	add x12, x12, x14
	b .L_m2_advance_common
.L_m2_advance_packc:
	add x12, x12, x24
.L_m2_advance_common:
	add x11, x11, x13
	sub w10, w10, w9
	b .L_m2_n_loop

.L_m2_done:
	RESTORE_REGS
	ret

	.macro ENTRY_W13 name, target, degree
	.global \name
	.type \name, %function
	.p2align 4
\name:
	mov w6, #\degree
	b \target
	.size \name, .-\name
	.endm

	.macro ENTRY_W13_ROWS name, target, degree
	.global \name
	.type \name, %function
	.p2align 4
\name:
	mov w6, #\degree
	b \target
	.size \name, .-\name
	.endm

	.macro ENTRY_W2 name, target
	.global \name
	.type \name, %function
	.p2align 4
\name:
	mov w6, #0
	b \target
	.size \name, .-\name
	.endm

	ENTRY_W13 moe_sve_w13_silu_poly4_packc_m12, moe_sve_kernel_m12, 4
	ENTRY_W13 moe_sve_w13_silu_poly5_packc_m12, moe_sve_kernel_m12, 5
	ENTRY_W13 moe_sve_w13_silu_poly6_packc_m12, moe_sve_kernel_m12, 6
	ENTRY_W13_ROWS moe_sve_w13_silu_poly4_packc_m12_rows, moe_sve_kernel_m12_rows, 4
	ENTRY_W13_ROWS moe_sve_w13_silu_poly5_packc_m12_rows, moe_sve_kernel_m12_rows, 5
	ENTRY_W13_ROWS moe_sve_w13_silu_poly6_packc_m12_rows, moe_sve_kernel_m12_rows, 6
	ENTRY_W13 moe_sve_w13_silu_poly4_packc,     moe_sve_kernel_m8, 4
	ENTRY_W13 moe_sve_w13_silu_poly5_packc,     moe_sve_kernel_m8, 5
	ENTRY_W13 moe_sve_w13_silu_poly6_packc,     moe_sve_kernel_m8, 6
	ENTRY_W13 moe_sve_w13_silu_poly4_packc_m4, moe_sve_kernel_m4, 4
	ENTRY_W13 moe_sve_w13_silu_poly5_packc_m4, moe_sve_kernel_m4, 5
	ENTRY_W13 moe_sve_w13_silu_poly6_packc_m4, moe_sve_kernel_m4, 6
	ENTRY_W13 moe_sve_w13_silu_poly4_packc_m2, moe_sve_kernel_m2, 4
	ENTRY_W13 moe_sve_w13_silu_poly5_packc_m2, moe_sve_kernel_m2, 5
	ENTRY_W13 moe_sve_w13_silu_poly6_packc_m2, moe_sve_kernel_m2, 6
	ENTRY_W13 moe_sve_w13_silu_poly4_packc_m1, moe_sve_kernel_m2, 4
	ENTRY_W13 moe_sve_w13_silu_poly5_packc_m1, moe_sve_kernel_m2, 5
	ENTRY_W13 moe_sve_w13_silu_poly6_packc_m1, moe_sve_kernel_m2, 6

	ENTRY_W2 moe_sve_w2_packed_m12, moe_sve_kernel_m12
	ENTRY_W2 moe_sve_w2_packed,     moe_sve_kernel_m8
	ENTRY_W2 moe_sve_w2_packed_m4,  moe_sve_kernel_m4
	ENTRY_W2 moe_sve_w2_packed_m2,  moe_sve_kernel_m2
	ENTRY_W2 moe_sve_w2_packed_m1,  moe_sve_kernel_m2

	.balign 16
.Lmoe_sve_silu_consts:
	.float 1.0
	.float 0.5
	.float 1.4426950408889634
	.float 0.6931471805599453
	.float 0.16666666
	.float 0.04166666
	.float 0.00833333
	.float 0.0013888889
	.float 87.0
	.float -87.0

	.section .note.GNU-stack,"",@progbits
