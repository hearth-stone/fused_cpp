// Standalone M12 load-policy and K-block experiments.
//
// These entry points intentionally duplicate only the production M12 BF16
// W2 body and epilogue. They are linked by bench_m12_streaming_b only and are
// not part of the fused-MoE extension or its dispatch table.

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

	// mode 0: LD1H, mode 1: two PLDL1STRM hints, mode 2: two
	// PLDL2STRM hints, mode 3: LDNT1H, mode 4: one PLDL1STRM hint,
	// mode 5: one PLDL2STRM hint, mode 6: one PLDL3STRM hint. The target
	// consumes one 64-byte B cache line per K4 body at its 128-bit runtime VL;
	// modes 4/5/6 isolate one hint per line.
	.macro M12_K4_BODY mode, ahead0=0, ahead1=0
	.if \mode == 1
	prfm pldl1strm, [x17, #\ahead0]
	prfm pldl1strm, [x17, #\ahead1]
	.elseif \mode == 2
	prfm pldl2strm, [x17, #\ahead0]
	prfm pldl2strm, [x17, #\ahead1]
	.elseif \mode == 4
	prfm pldl1strm, [x17, #\ahead0]
	.elseif \mode == 5
	prfm pldl2strm, [x17, #\ahead0]
	.elseif \mode == 6
	prfm pldl3strm, [x17, #\ahead0]
	.endif
	.if \mode == 3
	ldnt1h {z4.h}, p0/z, [x17]
	ldnt1h {z5.h}, p0/z, [x17, #1, mul vl]
	ldnt1h {z6.h}, p0/z, [x17, #2, mul vl]
	ldnt1h {z7.h}, p0/z, [x17, #3, mul vl]
	.else
	ld1h {z4.h}, p0/z, [x17]
	ld1h {z5.h}, p0/z, [x17, #1, mul vl]
	ld1h {z6.h}, p0/z, [x17, #2, mul vl]
	ld1h {z7.h}, p0/z, [x17, #3, mul vl]
	.endif
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

	.macro BUILD_C_BF16_OFFSETS_M12
	index z0.s, #0, #1
	mov z1.d, z0.d
	and z1.s, z1.s, #3
	mov z2.d, z0.d
	lsr z2.s, z2.s, #2
	lsl z2.s, z2.s, #4
	mov z3.d, z1.d
	and z3.s, z3.s, #1
	lsl z3.s, z3.s, #1
	add z0.s, z2.s, z3.s
	mov z2.d, z1.d
	lsr z2.s, z2.s, #1
	and z2.s, z2.s, #1
	dup z3.s, w25
	mul z2.s, p1/m, z2.s, z3.s
	add z0.s, z0.s, z2.s
	.endm

	.macro STORE_PAIR_W2_M12_BF16 base, a0, a1, a2, a3
	add x3, \base, #4
	add x4, \base, #8
	add x5, \base, #12
	bfcvt z4.h, p1/m, \a0\().s
	st1h {z4.s}, p1, [\base, z0.s, sxtw]
	bfcvt z4.h, p1/m, \a1\().s
	st1h {z4.s}, p1, [x3, z0.s, sxtw]
	bfcvt z4.h, p1/m, \a2\().s
	st1h {z4.s}, p1, [x4, z0.s, sxtw]
	bfcvt z4.h, p1/m, \a3\().s
	st1h {z4.s}, p1, [x5, z0.s, sxtw]
	.endm

	.macro STORE_W2_M12_BF16
	BUILD_C_BF16_OFFSETS_M12
	mov x16, x12
	STORE_PAIR_W2_M12_BF16 x16, z8, z9, z10, z11
	add x16, x12, x25, lsl #1
	STORE_PAIR_W2_M12_BF16 x16, z12, z13, z14, z15
	add x16, x12, x25, lsl #2
	STORE_PAIR_W2_M12_BF16 x16, z16, z17, z18, z19
	add x16, x16, x25, lsl #1
	STORE_PAIR_W2_M12_BF16 x16, z20, z21, z22, z23
	add x16, x12, x25, lsl #3
	STORE_PAIR_W2_M12_BF16 x16, z24, z25, z26, z27
	add x16, x16, x25, lsl #1
	STORE_PAIR_W2_M12_BF16 x16, z28, z29, z30, z31
	.endm

	.macro LOAD_PARTIAL_M12 base
	addvl x7, \base, #8
	addvl x8, \base, #16
	ld1w {z8.s}, p1/z, [\base]
	ld1w {z9.s}, p1/z, [\base, #1, mul vl]
	ld1w {z10.s}, p1/z, [\base, #2, mul vl]
	ld1w {z11.s}, p1/z, [\base, #3, mul vl]
	ld1w {z12.s}, p1/z, [\base, #4, mul vl]
	ld1w {z13.s}, p1/z, [\base, #5, mul vl]
	ld1w {z14.s}, p1/z, [\base, #6, mul vl]
	ld1w {z15.s}, p1/z, [\base, #7, mul vl]
	ld1w {z16.s}, p1/z, [x7]
	ld1w {z17.s}, p1/z, [x7, #1, mul vl]
	ld1w {z18.s}, p1/z, [x7, #2, mul vl]
	ld1w {z19.s}, p1/z, [x7, #3, mul vl]
	ld1w {z20.s}, p1/z, [x7, #4, mul vl]
	ld1w {z21.s}, p1/z, [x7, #5, mul vl]
	ld1w {z22.s}, p1/z, [x7, #6, mul vl]
	ld1w {z23.s}, p1/z, [x7, #7, mul vl]
	ld1w {z24.s}, p1/z, [x8]
	ld1w {z25.s}, p1/z, [x8, #1, mul vl]
	ld1w {z26.s}, p1/z, [x8, #2, mul vl]
	ld1w {z27.s}, p1/z, [x8, #3, mul vl]
	ld1w {z28.s}, p1/z, [x8, #4, mul vl]
	ld1w {z29.s}, p1/z, [x8, #5, mul vl]
	ld1w {z30.s}, p1/z, [x8, #6, mul vl]
	ld1w {z31.s}, p1/z, [x8, #7, mul vl]
	.endm

	.macro STORE_PARTIAL_M12 base
	addvl x7, \base, #8
	addvl x8, \base, #16
	st1w {z8.s}, p1, [\base]
	st1w {z9.s}, p1, [\base, #1, mul vl]
	st1w {z10.s}, p1, [\base, #2, mul vl]
	st1w {z11.s}, p1, [\base, #3, mul vl]
	st1w {z12.s}, p1, [\base, #4, mul vl]
	st1w {z13.s}, p1, [\base, #5, mul vl]
	st1w {z14.s}, p1, [\base, #6, mul vl]
	st1w {z15.s}, p1, [\base, #7, mul vl]
	st1w {z16.s}, p1, [x7]
	st1w {z17.s}, p1, [x7, #1, mul vl]
	st1w {z18.s}, p1, [x7, #2, mul vl]
	st1w {z19.s}, p1, [x7, #3, mul vl]
	st1w {z20.s}, p1, [x7, #4, mul vl]
	st1w {z21.s}, p1, [x7, #5, mul vl]
	st1w {z22.s}, p1, [x7, #6, mul vl]
	st1w {z23.s}, p1, [x7, #7, mul vl]
	st1w {z24.s}, p1, [x8]
	st1w {z25.s}, p1, [x8, #1, mul vl]
	st1w {z26.s}, p1, [x8, #2, mul vl]
	st1w {z27.s}, p1, [x8, #3, mul vl]
	st1w {z28.s}, p1, [x8, #4, mul vl]
	st1w {z29.s}, p1, [x8, #5, mul vl]
	st1w {z30.s}, p1, [x8, #6, mul vl]
	st1w {z31.s}, p1, [x8, #7, mul vl]
	.endm

	.macro DEFINE_M12_STREAM_KERNEL name, mode, ahead0=0, ahead1=0
	.global \name
	.type \name, %function
	.p2align 4
\name:
	SAVE_REGS
	ldp w19, w20, [x4]        // M=12, K
	ldr w21, [x4, #8]         // N
	ldr w23, [x4, #20]        // ldc
	mov x28, x0               // packed A
	cntb x27
	lsr x9, x27, #1           // N tile in BF16 columns
	lsl x25, x23, #1          // BF16 ldc in bytes
	mul x13, x20, x27         // packed-B bytes per N tile
	mov x14, x27              // row-major BF16 C advance per N tile
	ptrue p0.b
	ptrue p1.s
	mov x11, x1
	mov x12, x2
	mov w10, w21

.L_\name\()_n_loop:
	cmp w10, #0
	b.le .L_\name\()_done
	ZERO_ACC_M12
	mov x17, x11
	mov x18, x28
	mov w15, w20

.L_\name\()_k_loop:
	M12_K4_BODY \mode, \ahead0, \ahead1
	M12_K4_BODY \mode, \ahead0, \ahead1
	subs w15, w15, #8
	b.gt .L_\name\()_k_loop

	STORE_W2_M12_BF16
	add x12, x12, x14
	add x11, x11, x13
	sub w10, w10, w9
	b .L_\name\()_n_loop

.L_\name\()_done:
	RESTORE_REGS
	ret
	.size \name, .-\name
	.endm

	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_ld1h,            0
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl1strm_256,  1, 256,  320
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl1strm_512,  1, 512,  576
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl1strm_1024, 1, 1024, 1088
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl1strm_2048, 1, 2048, 2112
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl2strm_512,  2, 512,  576
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl2strm_1024, 2, 1024, 1088
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl2strm_2048, 2, 2048, 2112
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_ldnt1h,          3
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl1strm_2048_x1, 4, 2048
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl2strm_1024_x1, 5, 1024
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl3strm_512_x1,  6, 512
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl3strm_1024_x1, 6, 1024
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl3strm_2048_x1, 6, 2048
	DEFINE_M12_STREAM_KERNEL moe_sve_m12_bf16_pldl3strm_4096_x1, 6, 4096

	// Serial K-block experiment. x3 is a 12*N FP32 scratch buffer in
	// accumulator-native layout; w6 is the fixed wrapper's K chunk size.
	// w5 selects the packed-B layout: 0 keeps the production Ntile-major
	// layout, 1 uses Kchunk-major/Ntile-minor storage so each K chunk streams.
	.p2align 4
.L_moe_sve_m12_bf16_kblock:
	SAVE_REGS
	ldr w20, [x4, #4]         // K
	ldr w21, [x4, #8]         // N
	ldr w23, [x4, #20]        // ldc
	mov x29, x0               // packed-A base
	mov x19, x3               // partial-C base
	cntb x27
	lsr x9, x27, #1           // N tile in BF16 columns
	lsl x25, x23, #1          // BF16 ldc in bytes
	mov x14, x27              // BF16 C advance per N tile
	lsl x24, x27, #4
	add x24, x24, x27, lsl #3 // partial-C advance: 24 * VL
	ptrue p0.b
	ptrue p1.s
	mov w22, #0               // K chunk start

.L_m12_kblock_outer:
	sub w26, w20, w22
	cmp w26, w6
	csel w26, w26, w6, le     // min(K - k_start, Kc)
	uxtw x7, w22
	lsl x8, x7, #4
	add x8, x8, x7, lsl #3
	add x28, x29, x8          // A + k_start * 24 bytes
	cbnz w5, .L_m12_kblock_b_kblocked
	mul x13, x20, x27         // Ntile-major: bytes per full-K N tile
	mul x23, x7, x27          // Ntile-major: K-chunk offset in each N tile
	b .L_m12_kblock_b_layout_done

.L_m12_kblock_b_kblocked:
	uxtw x8, w21
	mul x23, x7, x8
	lsl x23, x23, #1          // Kchunk-major: bytes before this K chunk
	uxtw x8, w26
	mul x13, x8, x27          // Kchunk-major: bytes per N tile in this chunk

.L_m12_kblock_b_layout_done:
	mov x11, x1               // current packed-B N-tile base
	mov x12, x2               // current BF16 output N tile
	mov x16, x19              // current partial-C N tile
	mov w10, w21

.L_m12_kblock_n_loop:
	cmp w10, #0
	b.le .L_m12_kblock_next_k
	cmp w22, #0
	b.eq .L_m12_kblock_zero
	LOAD_PARTIAL_M12 x16
	b .L_m12_kblock_compute

.L_m12_kblock_zero:
	ZERO_ACC_M12

.L_m12_kblock_compute:
	add x17, x11, x23
	mov x18, x28
	mov w15, w26

.L_m12_kblock_k_loop:
	M12_K4_BODY 0
	M12_K4_BODY 0
	subs w15, w15, #8
	b.gt .L_m12_kblock_k_loop

	add w7, w22, w26
	cmp w7, w20
	b.eq .L_m12_kblock_final_store
	STORE_PARTIAL_M12 x16
	add x16, x16, x24
	b .L_m12_kblock_advance_n

.L_m12_kblock_final_store:
	add x7, x16, x24
	STORE_W2_M12_BF16
	mov x16, x7

.L_m12_kblock_advance_n:
	add x11, x11, x13
	add x12, x12, x14
	sub w10, w10, w9
	b .L_m12_kblock_n_loop

.L_m12_kblock_next_k:
	add w22, w22, w26
	cmp w22, w20
	b.lt .L_m12_kblock_outer
	RESTORE_REGS
	ret

	.macro DEFINE_M12_KBLOCK_ENTRY name, kc
	.global \name
	.type \name, %function
	.p2align 4
\name:
	mov w6, #\kc
	mov w5, #0
	b .L_moe_sve_m12_bf16_kblock
	.size \name, .-\name
	.endm

	DEFINE_M12_KBLOCK_ENTRY moe_sve_m12_bf16_kblock_256, 256
	DEFINE_M12_KBLOCK_ENTRY moe_sve_m12_bf16_kblock_512, 512
	DEFINE_M12_KBLOCK_ENTRY moe_sve_m12_bf16_kblock_768, 768
	DEFINE_M12_KBLOCK_ENTRY moe_sve_m12_bf16_kblock_1024, 1024

	.macro DEFINE_M12_KBLOCK_PACKED_ENTRY name, kc
	.global \name
	.type \name, %function
	.p2align 4
\name:
	mov w6, #\kc
	mov w5, #1
	b .L_moe_sve_m12_bf16_kblock
	.size \name, .-\name
	.endm

	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_256, 256
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_384, 384
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_448, 448
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_512, 512
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_544, 544
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_552, 552
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_560, 560
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_568, 568
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_576, 576
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_640, 640
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_704, 704
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_720, 720
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_736, 736
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_752, 752
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_768, 768
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_784, 784
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_792, 792
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_800, 800
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_816, 816
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_832, 832
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_848, 848
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_864, 864
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_880, 880
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_896, 896
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_960, 960
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_1024, 1024
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_1088, 1088
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_1152, 1152
	DEFINE_M12_KBLOCK_PACKED_ENTRY moe_sve_m12_bf16_kblock_packed_1280, 1280

	.section .note.GNU-stack,"",@progbits
