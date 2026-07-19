// Standalone M8/M4/M2/M1 assembly K-block experiment.
//
// The arithmetic pipelines and BF16 stores mirror the corresponding
// production kernels. Packed B is Kchunk-major/Ntile-minor. The benchmark
// appends a dynamic Kc at offset 24 of gemm_params_t.

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

	.macro BUILD_C_BF16_OFFSETS
	index z14.s, #0, #1
	mov z15.d, z14.d
	and z15.s, z15.s, #3
	mov z9.d, z14.d
	lsr z9.s, z9.s, #2
	lsl z9.s, z9.s, #4
	mov z13.d, z15.d
	and z13.s, z13.s, #1
	lsl z13.s, z13.s, #1
	add z9.s, z9.s, z13.s
	mov z12.d, z15.d
	lsr z12.s, z12.s, #1
	and z12.s, z12.s, #1
	cmpeq p4.s, p1/z, z12.s, #0
	dup z13.s, w25
	mul z12.s, p1/m, z12.s, z13.s
	add z9.s, z9.s, z12.s
	.endm

	.macro STORE_PAIR_BF16 base, pred, a0, a1, a2, a3
	add x3, \base, #4
	add x4, \base, #8
	add x5, \base, #12
	bfcvt z13.h, p1/m, \a0\().s
	st1h {z13.s}, \pred, [\base, z9.s, sxtw]
	bfcvt z13.h, p1/m, \a1\().s
	st1h {z13.s}, \pred, [x3, z9.s, sxtw]
	bfcvt z13.h, p1/m, \a2\().s
	st1h {z13.s}, \pred, [x4, z9.s, sxtw]
	bfcvt z13.h, p1/m, \a3\().s
	st1h {z13.s}, \pred, [x5, z9.s, sxtw]
	.endm

	.macro STORE_W2_PAIR_BF16 base, row, a0, a1, a2, a3
	cmp w19, #\row
	b.le 944\@f
	cmp w19, #(\row + 1)
	b.eq 945\@f
	STORE_PAIR_BF16 \base, p1, \a0, \a1, \a2, \a3
	b 944\@f
945\@:
	STORE_PAIR_BF16 \base, p4, \a0, \a1, \a2, \a3
944\@:
	.endm

	.macro STORE_W2_M8_BF16
	BUILD_C_BF16_OFFSETS
	mov x16, x12
	STORE_W2_PAIR_BF16 x16, 0, z16, z17, z18, z19
	add x16, x12, x25, lsl #1
	STORE_W2_PAIR_BF16 x16, 2, z20, z21, z22, z23
	add x16, x12, x25, lsl #2
	STORE_W2_PAIR_BF16 x16, 4, z24, z25, z26, z27
	add x16, x16, x25, lsl #1
	STORE_W2_PAIR_BF16 x16, 6, z28, z29, z30, z31
	.endm

	.macro STORE_W2_M4_BF16
	BUILD_C_BF16_OFFSETS
	mov x16, x12
	STORE_W2_PAIR_BF16 x16, 0, z16, z17, z18, z19
	add x16, x12, x25, lsl #1
	STORE_W2_PAIR_BF16 x16, 2, z20, z21, z22, z23
	.endm

	.macro STORE_W2_M2_BF16
	BUILD_C_BF16_OFFSETS
	mov x16, x12
	STORE_W2_PAIR_BF16 x16, 0, z16, z17, z18, z19
	.endm

	.macro LOAD_PARTIAL_M8 base
	addvl x7, \base, #8
	ld1w {z16.s}, p1/z, [\base]
	ld1w {z17.s}, p1/z, [\base, #1, mul vl]
	ld1w {z18.s}, p1/z, [\base, #2, mul vl]
	ld1w {z19.s}, p1/z, [\base, #3, mul vl]
	ld1w {z20.s}, p1/z, [\base, #4, mul vl]
	ld1w {z21.s}, p1/z, [\base, #5, mul vl]
	ld1w {z22.s}, p1/z, [\base, #6, mul vl]
	ld1w {z23.s}, p1/z, [\base, #7, mul vl]
	ld1w {z24.s}, p1/z, [x7]
	ld1w {z25.s}, p1/z, [x7, #1, mul vl]
	ld1w {z26.s}, p1/z, [x7, #2, mul vl]
	ld1w {z27.s}, p1/z, [x7, #3, mul vl]
	ld1w {z28.s}, p1/z, [x7, #4, mul vl]
	ld1w {z29.s}, p1/z, [x7, #5, mul vl]
	ld1w {z30.s}, p1/z, [x7, #6, mul vl]
	ld1w {z31.s}, p1/z, [x7, #7, mul vl]
	.endm

	.macro STORE_PARTIAL_M8 base
	addvl x7, \base, #8
	st1w {z16.s}, p1, [\base]
	st1w {z17.s}, p1, [\base, #1, mul vl]
	st1w {z18.s}, p1, [\base, #2, mul vl]
	st1w {z19.s}, p1, [\base, #3, mul vl]
	st1w {z20.s}, p1, [\base, #4, mul vl]
	st1w {z21.s}, p1, [\base, #5, mul vl]
	st1w {z22.s}, p1, [\base, #6, mul vl]
	st1w {z23.s}, p1, [\base, #7, mul vl]
	st1w {z24.s}, p1, [x7]
	st1w {z25.s}, p1, [x7, #1, mul vl]
	st1w {z26.s}, p1, [x7, #2, mul vl]
	st1w {z27.s}, p1, [x7, #3, mul vl]
	st1w {z28.s}, p1, [x7, #4, mul vl]
	st1w {z29.s}, p1, [x7, #5, mul vl]
	st1w {z30.s}, p1, [x7, #6, mul vl]
	st1w {z31.s}, p1, [x7, #7, mul vl]
	.endm

	.macro LOAD_PARTIAL_M4 base
	ld1w {z16.s}, p1/z, [\base]
	ld1w {z17.s}, p1/z, [\base, #1, mul vl]
	ld1w {z18.s}, p1/z, [\base, #2, mul vl]
	ld1w {z19.s}, p1/z, [\base, #3, mul vl]
	ld1w {z20.s}, p1/z, [\base, #4, mul vl]
	ld1w {z21.s}, p1/z, [\base, #5, mul vl]
	ld1w {z22.s}, p1/z, [\base, #6, mul vl]
	ld1w {z23.s}, p1/z, [\base, #7, mul vl]
	.endm

	.macro STORE_PARTIAL_M4 base
	st1w {z16.s}, p1, [\base]
	st1w {z17.s}, p1, [\base, #1, mul vl]
	st1w {z18.s}, p1, [\base, #2, mul vl]
	st1w {z19.s}, p1, [\base, #3, mul vl]
	st1w {z20.s}, p1, [\base, #4, mul vl]
	st1w {z21.s}, p1, [\base, #5, mul vl]
	st1w {z22.s}, p1, [\base, #6, mul vl]
	st1w {z23.s}, p1, [\base, #7, mul vl]
	.endm

	.macro LOAD_PARTIAL_M2 base
	ld1w {z16.s}, p1/z, [\base]
	ld1w {z17.s}, p1/z, [\base, #1, mul vl]
	ld1w {z18.s}, p1/z, [\base, #2, mul vl]
	ld1w {z19.s}, p1/z, [\base, #3, mul vl]
	.endm

	.macro STORE_PARTIAL_M2 base
	st1w {z16.s}, p1, [\base]
	st1w {z17.s}, p1, [\base, #1, mul vl]
	st1w {z18.s}, p1, [\base, #2, mul vl]
	st1w {z19.s}, p1, [\base, #3, mul vl]
	.endm

	.macro DEFINE_MSMALL_KBLOCK name, height
	.global \name
	.type \name, %function
	.p2align 4
\name:
	SAVE_REGS
	ldp w19, w20, [x4]        // logical M, K
	ldr w21, [x4, #8]         // N
	ldr w23, [x4, #20]        // ldc
	ldr w6, [x4, #24]         // experimental Kc
	mov x28, x0               // packed-A base
	mov x29, x3               // partial-C base
	cntb x27
	lsr x9, x27, #1           // N tile in BF16 columns
	lsl x25, x23, #1          // BF16 ldc in bytes
	mov x14, x27              // row-major BF16 C advance per N tile
	.if \height == 8
	lsl x24, x27, #4          // 16 accumulator vectors per N tile
	.elseif \height == 4
	lsl x24, x27, #3          // 8 accumulator vectors per N tile
	.else
	lsl x24, x27, #2          // 4 accumulator vectors per N tile
	.endif
	ptrue p0.b
	ptrue p1.s
	mov w22, #0               // K chunk start

.L_\name\()_outer:
	sub w26, w20, w22
	cmp w26, w6
	csel w26, w26, w6, le     // min(K - k_start, Kc)
	uxtw x7, w22
	lsl x8, x7, #4
	add x30, x28, x8          // packed A + k_start * 16 bytes
	uxtw x8, w21
	mul x23, x7, x8
	lsl x23, x23, #1          // bytes before this packed-B K chunk
	uxtw x8, w26
	mul x13, x8, x27          // packed-B bytes per N tile in this chunk
	mov x11, x1
	mov x12, x2
	mov x16, x29
	mov w10, w21

.L_\name\()_n_loop:
	cmp w10, #0
	b.le .L_\name\()_next_k
	cmp w22, #0
	b.eq .L_\name\()_zero
	.if \height == 8
	LOAD_PARTIAL_M8 x16
	.elseif \height == 4
	LOAD_PARTIAL_M4 x16
	.else
	LOAD_PARTIAL_M2 x16
	.endif
	b .L_\name\()_compute

.L_\name\()_zero:
	.if \height == 8
	ZERO_ACC_M8
	.elseif \height == 4
	ZERO_ACC_M4
	.else
	ZERO_ACC_M2
	.endif

.L_\name\()_compute:
	add x17, x11, x23
	mov x18, x30
	mov w15, w26

	.if \height == 8
	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_\name\()_tail_cur
.L_\name\()_k_loop:
	LOAD_AB_PACKED z8, z9, z10, z11, z12, z13, z14, z15
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_\name\()_tail_next
	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_\name\()_k_loop
.L_\name\()_tail_cur:
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	b .L_\name\()_computed
.L_\name\()_tail_next:
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15
	.elseif \height == 4
	LOAD_AB_PACKED_M4 z0, z1, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_\name\()_tail_cur
.L_\name\()_k_loop:
	LOAD_AB_PACKED_M4 z8, z9, z12, z13, z14, z15
	BFMMLA_M4_AB z0, z1, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_\name\()_tail_next
	LOAD_AB_PACKED_M4 z0, z1, z4, z5, z6, z7
	BFMMLA_M4_AB z8, z9, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_\name\()_k_loop
.L_\name\()_tail_cur:
	BFMMLA_M4_AB z0, z1, z4, z5, z6, z7
	b .L_\name\()_computed
.L_\name\()_tail_next:
	BFMMLA_M4_AB z8, z9, z12, z13, z14, z15
	.else
	LOAD_AB_PACKED_M2 z0, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_\name\()_tail_cur
.L_\name\()_k_loop:
	LOAD_AB_PACKED_M2 z8, z12, z13, z14, z15
	BFMMLA_M2_AB z0, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_\name\()_tail_next
	LOAD_AB_PACKED_M2 z0, z4, z5, z6, z7
	BFMMLA_M2_AB z8, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_\name\()_k_loop
.L_\name\()_tail_cur:
	BFMMLA_M2_AB z0, z4, z5, z6, z7
	b .L_\name\()_computed
.L_\name\()_tail_next:
	BFMMLA_M2_AB z8, z12, z13, z14, z15
	.endif

.L_\name\()_computed:
	add w7, w22, w26
	cmp w7, w20
	b.eq .L_\name\()_final_store
	.if \height == 8
	STORE_PARTIAL_M8 x16
	.elseif \height == 4
	STORE_PARTIAL_M4 x16
	.else
	STORE_PARTIAL_M2 x16
	.endif
	add x16, x16, x24
	b .L_\name\()_advance_n

.L_\name\()_final_store:
	add x7, x16, x24
	.if \height == 8
	STORE_W2_M8_BF16
	.elseif \height == 4
	STORE_W2_M4_BF16
	.else
	STORE_W2_M2_BF16
	.endif
	mov x16, x7

.L_\name\()_advance_n:
	add x11, x11, x13
	add x12, x12, x14
	sub w10, w10, w9
	b .L_\name\()_n_loop

.L_\name\()_next_k:
	add w22, w22, w26
	cmp w22, w20
	b.lt .L_\name\()_outer
	RESTORE_REGS
	ret
	.size \name, .-\name
	.endm

	DEFINE_MSMALL_KBLOCK moe_sve_m8_bf16_kblock_packed, 8
	DEFINE_MSMALL_KBLOCK moe_sve_m4_bf16_kblock_packed, 4
	DEFINE_MSMALL_KBLOCK moe_sve_m2_bf16_kblock_packed, 2
	DEFINE_MSMALL_KBLOCK moe_sve_m1_bf16_kblock_packed, 2

	.section .note.GNU-stack,"",@progbits
