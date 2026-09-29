// i8gemm_sve.S -- SVE I8 GEMM kernel entry points.
//
// ABI matches the NEON low-level kernels:
//   x0 A, x1 packed B, x2 C, x3 packed A or NULL, x4 params
// Bias variants additionally receive x5 = bias.
// Scaled variants receive x5 = row scale, x6 = column scale, x7 = bias/null.

	.text
	.arch armv8.6-a+sve+i8mm

	.macro SAVE_REGS
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp x23, x24, [sp, #-16]!
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
	ldp x23, x24, [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16
	.endm

	// Load two A rows (8 int8 each) into one replicated SVE vector.
	// row pair index is an immediate 0..3. Uses x7/x18 as temporaries.
	.macro LOAD_A_PAIR tag, zreg, rp
	mov x7, #(\rp * 2)
	cmp x7, x3
	b.ge .L_\tag\()_load_a_pair_\rp\()_zero
	add x7, x7, x15
	madd x7, x7, x11, x4
	mov x18, #(\rp * 2 + 1)
	cmp x18, x3
	b.ge .L_\tag\()_load_a_pair_\rp\()_zero_high
	add x18, x18, x15
	madd x18, x18, x11, x4
	ldr d0, [x0, x7]
	add x18, x0, x18
	ld1 {v0.d}[1], [x18]
	b .L_\tag\()_load_a_pair_\rp\()_dup
.L_\tag\()_load_a_pair_\rp\()_zero_high:
	ldr d0, [x0, x7]
	mov v0.d[1], xzr
.L_\tag\()_load_a_pair_\rp\()_dup:
	dup \zreg\().q, z0.q[0]
	b .L_\tag\()_load_a_pair_\rp\()_done
.L_\tag\()_load_a_pair_\rp\()_zero:
	mov \zreg\().b, #0
.L_\tag\()_load_a_pair_\rp\()_done:
	.endm

	.macro MMLA_ROW areg, c0, c1, c2, c3
	smmla \c0\().s, \areg\().b, z4.b
	smmla \c1\().s, \areg\().b, z5.b
	smmla \c2\().s, \areg\().b, z6.b
	smmla \c3\().s, \areg\().b, z7.b
	.endm

	.macro MMLA_BLOCK a0, a1, a2, a3, b0, b1, b2, b3
	smmla z16.s, \a0\().b, \b0\().b
	smmla z17.s, \a0\().b, \b1\().b
	smmla z18.s, \a0\().b, \b2\().b
	smmla z19.s, \a0\().b, \b3\().b
	smmla z20.s, \a1\().b, \b0\().b
	smmla z21.s, \a1\().b, \b1\().b
	smmla z22.s, \a1\().b, \b2\().b
	smmla z23.s, \a1\().b, \b3\().b
	smmla z24.s, \a2\().b, \b0\().b
	smmla z25.s, \a2\().b, \b1\().b
	smmla z26.s, \a2\().b, \b2\().b
	smmla z27.s, \a2\().b, \b3\().b
	smmla z28.s, \a3\().b, \b0\().b
	smmla z29.s, \a3\().b, \b1\().b
	smmla z30.s, \a3\().b, \b2\().b
	smmla z31.s, \a3\().b, \b3\().b
	.endm

	.macro LOAD_AB_PACKED_CUR
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	ld1rqb z0.b, p0/z, [x20]
	ld1rqb z1.b, p0/z, [x20, #16]
	ld1rqb z2.b, p0/z, [x20, #32]
	ld1rqb z3.b, p0/z, [x20, #48]
	add x20, x20, #64
	.endm

	.macro LOAD_AB_PACKED_NEXT
	ld1b z12.b, p0/z, [x6]
	ld1b z13.b, p0/z, [x6, #1, mul vl]
	ld1b z14.b, p0/z, [x6, #2, mul vl]
	ld1b z15.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	ld1rqb z8.b, p0/z, [x20]
	ld1rqb z9.b, p0/z, [x20, #16]
	ld1rqb z10.b, p0/z, [x20, #32]
	ld1rqb z11.b, p0/z, [x20, #48]
	add x20, x20, #64
	.endm

	// ACL-style software-pipelined K8 step: compute the current 8x2VL block
	// (A in z0..z3, B in z4..z7) while loading the next slice's B into
	// z12..z15 and next A into z8..z11. Loads are interleaved between the
	// smmla groups so the load units stay busy across the compute window and
	// the loaded data is ready by the next iteration.
	.macro MMLA_CUR_LDNEXT
	smmla z16.s, z0.b, z4.b
	smmla z17.s, z0.b, z5.b
	ld1b z12.b, p0/z, [x6]
	smmla z18.s, z0.b, z6.b
	smmla z19.s, z0.b, z7.b
	ld1b z13.b, p0/z, [x6, #1, mul vl]
	smmla z20.s, z1.b, z4.b
	smmla z21.s, z1.b, z5.b
	ld1b z14.b, p0/z, [x6, #2, mul vl]
	smmla z22.s, z1.b, z6.b
	smmla z23.s, z1.b, z7.b
	ld1b z15.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	smmla z24.s, z2.b, z4.b
	smmla z25.s, z2.b, z5.b
	ld1rqb z8.b, p0/z, [x20]
	smmla z26.s, z2.b, z6.b
	smmla z27.s, z2.b, z7.b
	ld1rqb z9.b, p0/z, [x20, #16]
	smmla z28.s, z3.b, z4.b
	smmla z29.s, z3.b, z5.b
	ld1rqb z10.b, p0/z, [x20, #32]
	smmla z30.s, z3.b, z6.b
	smmla z31.s, z3.b, z7.b
	ld1rqb z11.b, p0/z, [x20, #48]
	add x20, x20, #64
	prfm pldl1keep, [x6, #512]
	prfm pldl1keep, [x20, #256]
	.endm

	// Symmetric step: compute next block (A in z8..z11, B in z12..z15) while
	// loading the following slice's B into z4..z7 and A into z0..z3.
	.macro MMLA_NEXT_LDCUR
	smmla z16.s, z8.b, z12.b
	smmla z17.s, z8.b, z13.b
	ld1b z4.b, p0/z, [x6]
	smmla z18.s, z8.b, z14.b
	smmla z19.s, z8.b, z15.b
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	smmla z20.s, z9.b, z12.b
	smmla z21.s, z9.b, z13.b
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	smmla z22.s, z9.b, z14.b
	smmla z23.s, z9.b, z15.b
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	smmla z24.s, z10.b, z12.b
	smmla z25.s, z10.b, z13.b
	ld1rqb z0.b, p0/z, [x20]
	smmla z26.s, z10.b, z14.b
	smmla z27.s, z10.b, z15.b
	ld1rqb z1.b, p0/z, [x20, #16]
	smmla z28.s, z11.b, z12.b
	smmla z29.s, z11.b, z13.b
	ld1rqb z2.b, p0/z, [x20, #32]
	smmla z30.s, z11.b, z14.b
	smmla z31.s, z11.b, z15.b
	ld1rqb z3.b, p0/z, [x20, #48]
	add x20, x20, #64
	prfm pldl1keep, [x6, #512]
	prfm pldl1keep, [x20, #256]
	.endm

	.macro ZERO_ACC
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



	.macro MMLA_M12_4A
	smmla z8.s,  z0.b, z4.b
	smmla z12.s, z1.b, z4.b
	smmla z16.s, z2.b, z4.b
	smmla z20.s, z3.b, z4.b
	smmla z9.s,  z0.b, z5.b
	smmla z13.s, z1.b, z5.b
	smmla z17.s, z2.b, z5.b
	smmla z21.s, z3.b, z5.b
	smmla z10.s, z0.b, z6.b
	smmla z14.s, z1.b, z6.b
	smmla z18.s, z2.b, z6.b
	smmla z22.s, z3.b, z6.b
	smmla z11.s, z0.b, z7.b
	smmla z15.s, z1.b, z7.b
	smmla z19.s, z2.b, z7.b
	smmla z23.s, z3.b, z7.b
	.endm

	.macro MMLA_M12_2A
	smmla z24.s, z0.b, z4.b
	smmla z28.s, z1.b, z4.b
	smmla z25.s, z0.b, z5.b
	smmla z29.s, z1.b, z5.b
	smmla z26.s, z0.b, z6.b
	smmla z30.s, z1.b, z6.b
	smmla z27.s, z0.b, z7.b
	smmla z31.s, z1.b, z7.b
	.endm

	.macro M12_K8_BODY
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	ld1rqb z0.b, p0/z, [x20]
	ld1rqb z1.b, p0/z, [x20, #16]
	ld1rqb z2.b, p0/z, [x20, #32]
	ld1rqb z3.b, p0/z, [x20, #48]
	// Compute the rows 4-7 contributions first (z2/z3). Once those smmla
	prfm pldl1keep, [x6, #512]
	prfm pldl1keep, [x20, #384]
	// have issued, z2/z3 are free, so reload them with rows 8-11 and let the
	// load overlap the rows 0-3 compute below. This removes the load-use
	// stall the old "load rows 8-11 then immediately smmla" schedule had,
	// without needing extra architectural registers (24 accumulators leave
	// only z0-z7 for operands).
	smmla z16.s, z2.b, z4.b
	smmla z17.s, z2.b, z5.b
	smmla z18.s, z2.b, z6.b
	smmla z19.s, z2.b, z7.b
	smmla z20.s, z3.b, z4.b
	smmla z21.s, z3.b, z5.b
	smmla z22.s, z3.b, z6.b
	smmla z23.s, z3.b, z7.b
	ld1rqb z2.b, p0/z, [x20, #64]   // rows 8-9
	ld1rqb z3.b, p0/z, [x20, #80]   // rows 10-11
	add x20, x20, #96
	smmla z8.s,  z0.b, z4.b
	smmla z9.s,  z0.b, z5.b
	smmla z10.s, z0.b, z6.b
	smmla z11.s, z0.b, z7.b
	smmla z12.s, z1.b, z4.b
	smmla z13.s, z1.b, z5.b
	smmla z14.s, z1.b, z6.b
	smmla z15.s, z1.b, z7.b
	smmla z24.s, z2.b, z4.b
	smmla z25.s, z2.b, z5.b
	smmla z26.s, z2.b, z6.b
	smmla z27.s, z2.b, z7.b
	smmla z28.s, z3.b, z4.b
	smmla z29.s, z3.b, z5.b
	smmla z30.s, z3.b, z6.b
	smmla z31.s, z3.b, z7.b
	.endm

	// Build element offsets for one column pair:
	// lane pattern per 128b segment:
	//   r0/c0, r0/c1, r1/c0, r1/c1
	.macro BUILD_OFFSETS cp
	index z0.s, #0, #1
	lsr z1.s, z0.s, #2
	lsl z1.s, z1.s, #3
	mov z2.d, z0.d
	and z2.s, z2.s, #1
	add z1.s, z1.s, z2.s
	add z1.s, z1.s, #(\cp * 2)
	mov z2.d, z0.d
	and z2.s, z2.s, #2
	lsr z2.s, z2.s, #1
	dup z3.s, w12
	mul z2.s, p1/m, z2.s, z3.s
	add z2.s, z2.s, z1.s
	.endm

	.macro BUILD_I32_OFFSET_BASE out
	index \out\().s, #0, #1
	lsr \out\().s, \out\().s, #2
	lsl \out\().s, \out\().s, #3
	index z1.s, #0, #1
	and z1.s, z1.s, #1
	add \out\().s, \out\().s, z1.s
	index z1.s, #0, #1
	and z1.s, z1.s, #2
	lsr z1.s, z1.s, #1
	dup z2.s, w12
	mul z1.s, p1/m, z1.s, z2.s
	add \out\().s, \out\().s, z1.s
	.endm

	.macro BUILD_I32_STORE_OFFSETS
	BUILD_I32_OFFSET_BASE z8
	mov z9.d, z8.d
	add z9.s, z9.s, #2
	mov z10.d, z8.d
	add z10.s, z10.s, #4
	mov z11.d, z8.d
	add z11.s, z11.s, #6
	.endm

	.macro BUILD_I32_STORE_OFFSETS_M12
	BUILD_I32_OFFSET_BASE z0
	mov z1.d, z0.d
	add z1.s, z1.s, #2
	mov z2.d, z0.d
	add z2.s, z2.s, #4
	mov z3.d, z0.d
	add z3.s, z3.s, #6
	.endm


	// Build column offsets used by f32 stores once per N tile.
	// z8-z11: C offsets for cp0..cp3, z12-z15: bias offsets for cp0..cp3.
	.macro BUILD_F32_STORE_OFFSETS
	index z8.s, #0, #1
	lsr z12.s, z8.s, #2
	lsl z12.s, z12.s, #3
	mov z13.d, z8.d
	and z13.s, z13.s, #1
	add z12.s, z12.s, z13.s
	mov z13.d, z8.d
	and z13.s, z13.s, #2
	lsr z13.s, z13.s, #1
	dup z14.s, w12
	mul z13.s, p1/m, z13.s, z14.s
	add z8.s, z13.s, z12.s
	mov z9.d, z8.d
	add z9.s, z9.s, #2
	mov z10.d, z8.d
	add z10.s, z10.s, #4
	mov z11.d, z8.d
	add z11.s, z11.s, #6
	mov z13.d, z12.d
	add z13.s, z13.s, #2
	mov z14.d, z12.d
	add z14.s, z14.s, #4
	mov z15.d, z12.d
	add z15.s, z15.s, #6
	.endm

	.macro LOAD_C_PAIR acc, cp, pred
	BUILD_OFFSETS \cp
	ld1w \acc\().s, \pred/z, [x18, z2.s, sxtw #2]
	.endm

	.macro STORE_PAIR_I32 acc, cp, pred
	BUILD_OFFSETS \cp
	st1w \acc\().s, \pred, [x18, z2.s, sxtw #2]
	.endm

	.macro STORE_PAIR_F32_OFF acc, off, pred, tmp
	scvtf \tmp\().s, p1/m, \acc\().s
	st1w \tmp\().s, \pred, [x18, \off\().s, sxtw #2]
	.endm

	.macro STORE_PAIR_F32_BIAS_OFF acc, coff, boff, pred, tmp, btmp
	scvtf \tmp\().s, p1/m, \acc\().s
	ld1w \btmp\().s, p1/z, [x6, \boff\().s, sxtw #2]
	fadd \tmp\().s, \tmp\().s, \btmp\().s
	st1w \tmp\().s, \pred, [x18, \coff\().s, sxtw #2]
	.endm

	.macro LOAD_C_ROW tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_load_c_row_\rp\()_zero
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_load_c_row_\rp\()_one
	LOAD_C_PAIR \a0, 0, p1
	LOAD_C_PAIR \a1, 1, p1
	LOAD_C_PAIR \a2, 2, p1
	LOAD_C_PAIR \a3, 3, p1
	b .L_\tag\()_load_c_row_\rp\()_done
.L_\tag\()_load_c_row_\rp\()_one:
	LOAD_C_PAIR \a0, 0, p4
	LOAD_C_PAIR \a1, 1, p4
	LOAD_C_PAIR \a2, 2, p4
	LOAD_C_PAIR \a3, 3, p4
	b .L_\tag\()_load_c_row_\rp\()_done
.L_\tag\()_load_c_row_\rp\()_zero:
	mov \a0\().s, #0
	mov \a1\().s, #0
	mov \a2\().s, #0
	mov \a3\().s, #0
.L_\tag\()_load_c_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_I32 tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_i32_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_i32_row_\rp\()_one
	STORE_PAIR_I32 \a0, 0, p1
	STORE_PAIR_I32 \a1, 1, p1
	STORE_PAIR_I32 \a2, 2, p1
	STORE_PAIR_I32 \a3, 3, p1
	b .L_\tag\()_store_i32_row_\rp\()_done
.L_\tag\()_store_i32_row_\rp\()_one:
	STORE_PAIR_I32 \a0, 0, p4
	STORE_PAIR_I32 \a1, 1, p4
	STORE_PAIR_I32 \a2, 2, p4
	STORE_PAIR_I32 \a3, 3, p4
.L_\tag\()_store_i32_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_F32 tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_f32_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_f32_row_\rp\()_one
	STORE_PAIR_F32_OFF \a0, z8, p1, z4
	STORE_PAIR_F32_OFF \a1, z9, p1, z5
	STORE_PAIR_F32_OFF \a2, z10, p1, z6
	STORE_PAIR_F32_OFF \a3, z11, p1, z7
	b .L_\tag\()_store_f32_row_\rp\()_done
.L_\tag\()_store_f32_row_\rp\()_one:
	STORE_PAIR_F32_OFF \a0, z8, p4, z4
	STORE_PAIR_F32_OFF \a1, z9, p4, z5
	STORE_PAIR_F32_OFF \a2, z10, p4, z6
	STORE_PAIR_F32_OFF \a3, z11, p4, z7
.L_\tag\()_store_f32_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_F32_BIAS tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_bias_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_bias_row_\rp\()_one
	STORE_PAIR_F32_BIAS_OFF \a0, z8, z12, p1, z4, z0
	STORE_PAIR_F32_BIAS_OFF \a1, z9, z13, p1, z5, z1
	STORE_PAIR_F32_BIAS_OFF \a2, z10, z14, p1, z6, z2
	STORE_PAIR_F32_BIAS_OFF \a3, z11, z15, p1, z7, z3
	b .L_\tag\()_store_bias_row_\rp\()_done
.L_\tag\()_store_bias_row_\rp\()_one:
	STORE_PAIR_F32_BIAS_OFF \a0, z8, z12, p4, z4, z0
	STORE_PAIR_F32_BIAS_OFF \a1, z9, z13, p4, z5, z1
	STORE_PAIR_F32_BIAS_OFF \a2, z10, z14, p4, z6, z2
	STORE_PAIR_F32_BIAS_OFF \a3, z11, z15, p4, z7, z3
.L_\tag\()_store_bias_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_I32_FAST tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_i32_fast_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_i32_fast_row_\rp\()_one
	st1w \a0\().s, p1, [x18, z8.s, sxtw #2]
	st1w \a1\().s, p1, [x18, z9.s, sxtw #2]
	st1w \a2\().s, p1, [x18, z10.s, sxtw #2]
	st1w \a3\().s, p1, [x18, z11.s, sxtw #2]
	b .L_\tag\()_store_i32_fast_row_\rp\()_done
.L_\tag\()_store_i32_fast_row_\rp\()_one:
	st1w \a0\().s, p4, [x18, z8.s, sxtw #2]
	st1w \a1\().s, p4, [x18, z9.s, sxtw #2]
	st1w \a2\().s, p4, [x18, z10.s, sxtw #2]
	st1w \a3\().s, p4, [x18, z11.s, sxtw #2]
.L_\tag\()_store_i32_fast_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	// Emit two quads at VL128 or four at VL256 for one deinterleaved row.
	// z0=[c0-3|c8-11], z1=[c4-7|c12-15] at VL256; x7 = row byte addr.
	// p3 masks the high quads when the N tile has only 8 columns.
	// mode: i32 (raw), f32 (scvtf), bias (scvtf + per-col bias in z8..z11).
	.macro DEINT_QUADS mode
	.ifnc \mode, i32
	scvtf z0.s, p1/m, z0.s
	scvtf z1.s, p1/m, z1.s
	.endif
	mov z4.d, z0.d
	ext z4.b, z4.b, z4.b, #16
	mov z5.d, z1.d
	ext z5.b, z5.b, z5.b, #16
	.ifc \mode, bias
	fadd z0.s, z0.s, z8.s
	fadd z1.s, z1.s, z9.s
	fadd z4.s, z4.s, z10.s
	fadd z5.s, z5.s, z11.s
	.endif
	st1w z0.s, p2, [x7]
	add x7, x7, #16
	st1w z1.s, p2, [x7]
	add x7, x7, #16
	st1w z4.s, p3, [x7]
	add x7, x7, #16
	st1w z5.s, p3, [x7]
	.endm

	// Deinterleave-store one row-pair (rows 2*rp, 2*rp+1) of the 8xN tile
	// using contiguous st1w instead of scatter. The smmla 2x2 output is laid
	// out (per accumulator) as [r0c0,r0c1,r1c0,r1c1 | r0c8,r0c9,r1c8,r1c9].
	// trn1.d gathers the even (r0) col pairs, trn2.d the odd (r1); ext brings
	// the high quad (cols 8-15) down so each store is a contiguous quad.
	// x18 = tile base byte addr (C[m0][n0]); x12 = ldc; cnt = row count;
	// p2 = VL4; x4/x7 scratch (x5 preserved for bias base).
	.macro DEINT_STORE_PAIR mode, tag, cnt, rp, a0, a1, a2, a3
	cmp \cnt, #(\rp * 2)
	b.le .L_\tag\()_deint_\rp\()_done
	trn1 z0.d, \a0\().d, \a1\().d
	trn1 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2)
	mul x7, x4, x12
	add x7, x18, x7, lsl #2
	DEINT_QUADS \mode
	cmp \cnt, #(\rp * 2 + 1)
	b.le .L_\tag\()_deint_\rp\()_done
	trn2 z0.d, \a0\().d, \a1\().d
	trn2 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2 + 1)
	mul x7, x4, x12
	add x7, x18, x7, lsl #2
	DEINT_QUADS \mode
.L_\tag\()_deint_\rp\()_done:
	.endm

	// Load the four contiguous channel-scale quads and optional bias quads for
	// the current 2VL N tile. The accumulator registers are dead at this point,
	// so z8-z15 are available to the direct-output epilogue.
	.macro LOAD_SCALED_TILE_AUX tag
	add x5, x22, x16, lsl #2
	ld1w z8.s, p2/z, [x5]
	add x5, x5, #16
	ld1w z9.s, p2/z, [x5]
	add x5, x5, #16
	ld1w z10.s, p3/z, [x5]
	add x5, x5, #16
	ld1w z11.s, p3/z, [x5]
	cbz x23, .L_\tag\()_scaled_no_bias
	add x5, x23, x16, lsl #2
	ld1w z12.s, p2/z, [x5]
	add x5, x5, #16
	ld1w z13.s, p2/z, [x5]
	add x5, x5, #16
	ld1w z14.s, p3/z, [x5]
	add x5, x5, #16
	ld1w z15.s, p3/z, [x5]
	b .L_\tag\()_scaled_aux_done
.L_\tag\()_scaled_no_bias:
	mov z12.s, #0
	mov z13.s, #0
	mov z14.s, #0
	mov z15.s, #0
.L_\tag\()_scaled_aux_done:
	mov w5, #0x7fff
	dup z7.s, w5
	.endm

	.macro SCALE_QUAD value, column_scale, bias
	scvtf \value\().s, p1/m, \value\().s
	fmul \value\().s, \value\().s, z6.s
	fmul \value\().s, \value\().s, \column_scale\().s
	fadd \value\().s, \value\().s, \bias\().s
	.endm

	// z0/z1 contain one deinterleaved row. Scale all quads before reusing z6
	// as the BF16 rounding scratch; z7 holds 0x7fff for round-to-nearest-even.
	.macro DEINT_SCALED_QUADS mode
	mov z4.d, z0.d
	ext z4.b, z4.b, z4.b, #16
	mov z5.d, z1.d
	ext z5.b, z5.b, z5.b, #16
	SCALE_QUAD z0, z8, z12
	SCALE_QUAD z1, z9, z13
	SCALE_QUAD z4, z10, z14
	SCALE_QUAD z5, z11, z15
	.ifc \mode, f32
	st1w z0.s, p2, [x7]
	add x7, x7, #16
	st1w z1.s, p2, [x7]
	add x7, x7, #16
	st1w z4.s, p3, [x7]
	add x7, x7, #16
	st1w z5.s, p3, [x7]
	.else
	lsr z6.s, z0.s, #16
	and z6.s, z6.s, #1
	add z6.s, z6.s, z7.s
	add z0.s, z0.s, z6.s
	lsr z0.s, z0.s, #16
	st1h z0.s, p2, [x7]
	add x7, x7, #8
	lsr z6.s, z1.s, #16
	and z6.s, z6.s, #1
	add z6.s, z6.s, z7.s
	add z1.s, z1.s, z6.s
	lsr z1.s, z1.s, #16
	st1h z1.s, p2, [x7]
	add x7, x7, #8
	lsr z6.s, z4.s, #16
	and z6.s, z6.s, #1
	add z6.s, z6.s, z7.s
	add z4.s, z4.s, z6.s
	lsr z4.s, z4.s, #16
	st1h z4.s, p3, [x7]
	add x7, x7, #8
	lsr z6.s, z5.s, #16
	and z6.s, z6.s, #1
	add z6.s, z6.s, z7.s
	add z5.s, z5.s, z6.s
	lsr z5.s, z5.s, #16
	st1h z5.s, p3, [x7]
	.endif
	.endm

	.macro DEINT_STORE_PAIR_SCALED mode, tag, cnt, rp, shift, a0, a1, a2, a3
	cmp \cnt, #(\rp * 2)
	b.le .L_\tag\()_scaled_deint_\rp\()_done
	trn1 z0.d, \a0\().d, \a1\().d
	trn1 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2)
	mul x7, x4, x12
	add x7, x18, x7, lsl #\shift
	add x5, x15, x4
	add x5, x21, x5, lsl #2
	ld1rw z6.s, p1/z, [x5]
	DEINT_SCALED_QUADS \mode
	cmp \cnt, #(\rp * 2 + 1)
	b.le .L_\tag\()_scaled_deint_\rp\()_done
	trn2 z0.d, \a0\().d, \a1\().d
	trn2 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2 + 1)
	mul x7, x4, x12
	add x7, x18, x7, lsl #\shift
	add x5, x15, x4
	add x5, x21, x5, lsl #2
	ld1rw z6.s, p1/z, [x5]
	DEINT_SCALED_QUADS \mode
.L_\tag\()_scaled_deint_\rp\()_done:
	.endm

	.macro STORE_C_SCALED_F32 tag
	LOAD_SCALED_TILE_AUX \tag
	DEINT_STORE_PAIR_SCALED f32, \tag, x3, 0, 2, z16, z17, z18, z19
	DEINT_STORE_PAIR_SCALED f32, \tag, x3, 1, 2, z20, z21, z22, z23
	DEINT_STORE_PAIR_SCALED f32, \tag, x3, 2, 2, z24, z25, z26, z27
	DEINT_STORE_PAIR_SCALED f32, \tag, x3, 3, 2, z28, z29, z30, z31
	.endm

	.macro STORE_C_SCALED_BF16 tag
	LOAD_SCALED_TILE_AUX \tag
	DEINT_STORE_PAIR_SCALED bf16, \tag, x3, 0, 1, z16, z17, z18, z19
	DEINT_STORE_PAIR_SCALED bf16, \tag, x3, 1, 1, z20, z21, z22, z23
	DEINT_STORE_PAIR_SCALED bf16, \tag, x3, 2, 1, z24, z25, z26, z27
	DEINT_STORE_PAIR_SCALED bf16, \tag, x3, 3, 1, z28, z29, z30, z31
	.endm

	.macro STORE_ROW_I32_M12 rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x8, #(\rp * 2)
	b.le .L_store_i32_m12_row_\rp\()_done
	cmp x8, #(\rp * 2 + 1)
	b.le .L_store_i32_m12_row_\rp\()_one
	st1w \a0\().s, p1, [x18, z0.s, sxtw #2]
	st1w \a1\().s, p1, [x18, z1.s, sxtw #2]
	st1w \a2\().s, p1, [x18, z2.s, sxtw #2]
	st1w \a3\().s, p1, [x18, z3.s, sxtw #2]
	b .L_store_i32_m12_row_\rp\()_done
.L_store_i32_m12_row_\rp\()_one:
	st1w \a0\().s, p4, [x18, z0.s, sxtw #2]
	st1w \a1\().s, p4, [x18, z1.s, sxtw #2]
	st1w \a2\().s, p4, [x18, z2.s, sxtw #2]
	st1w \a3\().s, p4, [x18, z3.s, sxtw #2]
.L_store_i32_m12_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro M12_SCALE_STORE_QUAD mode, tag, rp, half, cp, value, pred
	add x4, x22, x16, lsl #2
	add x4, x4, #(\cp * 16)
	ld1w z2.s, \pred/z, [x4]
	scvtf \value\().s, p1/m, \value\().s
	fmul \value\().s, \value\().s, z6.s
	fmul \value\().s, \value\().s, z2.s
	cbz x23, .L_\tag\()_m12_no_bias_\rp\()_\half\()_\cp
	add x4, x23, x16, lsl #2
	add x4, x4, #(\cp * 16)
	ld1w z3.s, \pred/z, [x4]
	fadd \value\().s, \value\().s, z3.s
.L_\tag\()_m12_no_bias_\rp\()_\half\()_\cp:
	.ifc \mode, f32
	st1w \value\().s, \pred, [x7]
	add x7, x7, #16
	.else
	lsr z2.s, \value\().s, #16
	and z2.s, z2.s, #1
	mov w4, #0x7fff
	dup z3.s, w4
	add z2.s, z2.s, z3.s
	add \value\().s, \value\().s, z2.s
	lsr \value\().s, \value\().s, #16
	st1h \value\().s, \pred, [x7]
	add x7, x7, #8
	.endif
	.endm

	.macro M12_DEINT_STORE_PAIR_SCALED mode, tag, rp, shift, a0, a1, a2, a3
	trn1 z0.d, \a0\().d, \a1\().d
	trn1 z1.d, \a2\().d, \a3\().d
	mov z4.d, z0.d
	ext z4.b, z4.b, z4.b, #16
	mov z5.d, z1.d
	ext z5.b, z5.b, z5.b, #16
	mov x4, #(\rp * 2)
	mul x7, x4, x12
	add x7, x18, x7, lsl #\shift
	add x4, x21, x4, lsl #2
	ld1rw z6.s, p1/z, [x4]
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 0, 0, z0, p2
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 0, 1, z1, p2
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 0, 2, z4, p3
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 0, 3, z5, p3
	trn2 z0.d, \a0\().d, \a1\().d
	trn2 z1.d, \a2\().d, \a3\().d
	mov z4.d, z0.d
	ext z4.b, z4.b, z4.b, #16
	mov z5.d, z1.d
	ext z5.b, z5.b, z5.b, #16
	mov x4, #(\rp * 2 + 1)
	mul x7, x4, x12
	add x7, x18, x7, lsl #\shift
	add x4, x21, x4, lsl #2
	ld1rw z6.s, p1/z, [x4]
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 1, 0, z0, p2
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 1, 1, z1, p2
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 1, 2, z4, p3
	M12_SCALE_STORE_QUAD \mode, \tag, \rp, 1, 3, z5, p3
	.endm



	.global i8gemm_k_nld_m12
	.type i8gemm_k_nld_m12, %function
i8gemm_k_nld_m12:
	SAVE_REGS
	mov x19, x3               // packed-A base

	ldr w8, [x4, #0]          // M, normally <= 12
	ldr w9, [x4, #4]          // K
	ldr w10, [x4, #8]         // N
	ldr w12, [x4, #20]        // ldc

	ptrue p0.b
	ptrue p1.s
	ptrue p2.s, vl4
	cntw x4
	sub x4, x4, #4
	whilelt p3.s, xzr, x4     // four high-quad lanes at VL256, none at VL128
	index z0.s, #0, #1
	and z0.s, z0.s, #2
	cmpeq p4.s, p1/z, z0.s, #0

	cntb x13
	lsr x13, x13, #1         // n_tile = VL bytes / 2
	mul x14, x9, x13         // packed-B N-tile stride in bytes

	mov x16, #0              // n0
	mov x17, x1              // current B tile
.L_m12_n_loop:
	mov x18, x16
	add x18, x2, x18, lsl #2
	ZERO_ACC_M12

	mov x4, x9
	mov x6, x17
	mov x20, x19
.L_m12_k_loop:
	M12_K8_BODY
	M12_K8_BODY
	subs x4, x4, #16
	b.gt .L_m12_k_loop

	mov x18, x16
	add x18, x2, x18, lsl #2
	DEINT_STORE_PAIR i32, m12, x8, 0, z8, z9, z10, z11
	DEINT_STORE_PAIR i32, m12, x8, 1, z12, z13, z14, z15
	DEINT_STORE_PAIR i32, m12, x8, 2, z16, z17, z18, z19
	DEINT_STORE_PAIR i32, m12, x8, 3, z20, z21, z22, z23
	DEINT_STORE_PAIR i32, m12, x8, 4, z24, z25, z26, z27
	DEINT_STORE_PAIR i32, m12, x8, 5, z28, z29, z30, z31

	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .L_m12_n_loop

	RESTORE_REGS
	ret
	.size i8gemm_k_nld_m12, .-i8gemm_k_nld_m12

	.macro M12_SCALED_BODY func_name, mode, c_shift
	.global \func_name
	.type \func_name, %function
\func_name:
	SAVE_REGS
	mov x19, x3
	mov x21, x5
	mov x22, x6
	mov x23, x7
	ldr w8, [x4, #0]
	ldr w9, [x4, #4]
	ldr w10, [x4, #8]
	ldr w12, [x4, #20]
	ptrue p0.b
	ptrue p1.s
	ptrue p2.s, vl4
	cntw x4
	sub x4, x4, #4
	whilelt p3.s, xzr, x4
	cntb x13
	lsr x13, x13, #1
	mul x14, x9, x13
	mov x16, #0
	mov x17, x1
.L_\func_name\()_n_loop:
	mov x18, x16
	add x18, x2, x18, lsl #\c_shift
	ZERO_ACC_M12
	mov x4, x9
	mov x6, x17
	mov x20, x19
.L_\func_name\()_k_loop:
	M12_K8_BODY
	M12_K8_BODY
	subs x4, x4, #16
	b.gt .L_\func_name\()_k_loop
	mov x18, x16
	add x18, x2, x18, lsl #\c_shift
	M12_DEINT_STORE_PAIR_SCALED \mode, \func_name, 0, \c_shift, z8, z9, z10, z11
	M12_DEINT_STORE_PAIR_SCALED \mode, \func_name, 1, \c_shift, z12, z13, z14, z15
	M12_DEINT_STORE_PAIR_SCALED \mode, \func_name, 2, \c_shift, z16, z17, z18, z19
	M12_DEINT_STORE_PAIR_SCALED \mode, \func_name, 3, \c_shift, z20, z21, z22, z23
	M12_DEINT_STORE_PAIR_SCALED \mode, \func_name, 4, \c_shift, z24, z25, z26, z27
	M12_DEINT_STORE_PAIR_SCALED \mode, \func_name, 5, \c_shift, z28, z29, z30, z31
	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .L_\func_name\()_n_loop
	RESTORE_REGS
	ret
	.size \func_name, .-\func_name
	.endm

	M12_SCALED_BODY i8gemm_k_scaled_m12_f, f32, 2
	M12_SCALED_BODY i8gemm_k_scaled_m12_b, bf16, 1

	.macro LOAD_C_ACC tag
	LOAD_C_ROW \tag, 0, z16, z17, z18, z19
	LOAD_C_ROW \tag, 1, z20, z21, z22, z23
	LOAD_C_ROW \tag, 2, z24, z25, z26, z27
	LOAD_C_ROW \tag, 3, z28, z29, z30, z31
	.endm

	.macro LOAD_C_ZERO tag
	ZERO_ACC
	.endm

	.macro STORE_C_I32 tag
	DEINT_STORE_PAIR i32, \tag, x3, 0, z16, z17, z18, z19
	DEINT_STORE_PAIR i32, \tag, x3, 1, z20, z21, z22, z23
	DEINT_STORE_PAIR i32, \tag, x3, 2, z24, z25, z26, z27
	DEINT_STORE_PAIR i32, \tag, x3, 3, z28, z29, z30, z31
	.endm

	.macro STORE_C_F32 tag
	DEINT_STORE_PAIR f32, \tag, x3, 0, z16, z17, z18, z19
	DEINT_STORE_PAIR f32, \tag, x3, 1, z20, z21, z22, z23
	DEINT_STORE_PAIR f32, \tag, x3, 2, z24, z25, z26, z27
	DEINT_STORE_PAIR f32, \tag, x3, 3, z28, z29, z30, z31
	.endm

	.macro STORE_C_F32_BIAS tag
	add x6, x5, x16, lsl #2          // bias base for this n-tile (cols n0..)
	ld1w {z8.s}, p2/z, [x6]          // bias c0-3
	add x6, x6, #16
	ld1w {z9.s}, p2/z, [x6]          // bias c4-7
	add x6, x6, #16
	ld1w {z10.s}, p3/z, [x6]         // bias c8-11
	add x6, x6, #16
	ld1w {z11.s}, p3/z, [x6]         // bias c12-15
	DEINT_STORE_PAIR bias, \tag, x3, 0, z16, z17, z18, z19
	DEINT_STORE_PAIR bias, \tag, x3, 1, z20, z21, z22, z23
	DEINT_STORE_PAIR bias, \tag, x3, 2, z24, z25, z26, z27
	DEINT_STORE_PAIR bias, \tag, x3, 3, z28, z29, z30, z31
	.endm

	.macro GEMM_BODY func_name, load_c, store_c, c_shift=2, scaled=0
	.global \func_name
	.type \func_name, %function
\func_name:
	SAVE_REGS
	mov x19, x3               // optional packed-A base, NULL for gather path
	.if \scaled
	mov x21, x5               // per-row activation scales
	mov x22, x6               // padded per-column weight scales
	mov x23, x7               // padded bias or NULL
	.endif

	ldr w8, [x4, #0]          // M
	ldr w9, [x4, #4]          // K
	ldr w10, [x4, #8]         // N
	ldr w11, [x4, #12]        // lda
	ldr w12, [x4, #20]        // ldc

	ptrue p0.b
	ptrue p1.s
	ptrue p2.s, vl4           // first 4 .s lanes (one 128-bit quad) for deint stores
	cntw x4
	sub x4, x4, #4
	whilelt p3.s, xzr, x4

	index z0.s, #0, #1
	and z0.s, z0.s, #2
	cmpeq p4.s, p1/z, z0.s, #0

	cntb x13
	lsr x13, x13, #1         // n_tile = VL bytes / 2
	mul x14, x9, x13         // B packed tile stride in bytes

	mov x15, #0              // m0
.L_\func_name\()_m_loop:
	cmp x15, x8
	b.ge .L_\func_name\()_done
	sub x3, x8, x15
	cmp x3, #8
	mov x18, #8
	csel x3, x3, x18, le     // m_now = min(M - m0, 8)

	mov x16, #0              // n0
	mov x17, x1              // current B tile
.L_\func_name\()_n_loop:
	// C base for this M/N tile, in bytes.
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #\c_shift

	\load_c \func_name

	cbnz x19, .L_\func_name\()_k_loop_packed_entry
	mov x4, #0               // kb
	mov x6, x17              // B K pointer
.L_\func_name\()_k_loop:
	cmp x4, x9
	b.ge .L_\func_name\()_store

	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]

	LOAD_A_PAIR \func_name, z0, 0
	MMLA_ROW z0, z16, z17, z18, z19
	LOAD_A_PAIR \func_name, z0, 1
	MMLA_ROW z0, z20, z21, z22, z23
	LOAD_A_PAIR \func_name, z0, 2
	MMLA_ROW z0, z24, z25, z26, z27
	LOAD_A_PAIR \func_name, z0, 3
	MMLA_ROW z0, z28, z29, z30, z31

	add x4, x4, #8
	addvl x6, x6, #4
	b .L_\func_name\()_k_loop

.L_\func_name\()_k_loop_packed_entry:
	mov x4, #0
	mov x6, x17              // B K pointer
	lsr x20, x15, #3
	mul x20, x20, x9
	add x20, x19, x20, lsl #3

	cmp x4, x9
	b.ge .L_\func_name\()_store
	LOAD_AB_PACKED_CUR
	add x4, x4, #8
	cmp x4, x9
	b.ge .L_\func_name\()_packed_tail_cur

.L_\func_name\()_k_loop_packed:
	MMLA_CUR_LDNEXT
	add x4, x4, #8
	cmp x4, x9
	b.ge .L_\func_name\()_packed_tail_next
	MMLA_NEXT_LDCUR
	add x4, x4, #8
	cmp x4, x9
	b.lt .L_\func_name\()_k_loop_packed

.L_\func_name\()_packed_tail_cur:
	MMLA_BLOCK z0, z1, z2, z3, z4, z5, z6, z7
	b .L_\func_name\()_store

.L_\func_name\()_packed_tail_next:
	MMLA_BLOCK z8, z9, z10, z11, z12, z13, z14, z15
	b .L_\func_name\()_store

.L_\func_name\()_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #\c_shift
	\store_c \func_name

	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .L_\func_name\()_n_loop

	add x15, x15, #8
	b .L_\func_name\()_m_loop

.L_\func_name\()_done:
	RESTORE_REGS
	ret
	.size \func_name, .-\func_name
	.endm

	GEMM_BODY i8gemm_k_ld, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld, LOAD_C_ZERO, STORE_C_I32
	GEMM_BODY i8gemm_k_ld1, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld1, LOAD_C_ZERO, STORE_C_I32
	GEMM_BODY i8gemm_k_ld2, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld2, LOAD_C_ZERO, STORE_C_I32
	GEMM_BODY i8gemm_k_ld4, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld4, LOAD_C_ZERO, STORE_C_I32

	GEMM_BODY i8gemm_k_ld_f, LOAD_C_ZERO, STORE_C_F32
	GEMM_BODY i8gemm_k_ld1_f, LOAD_C_ZERO, STORE_C_F32
	GEMM_BODY i8gemm_k_ld2_f, LOAD_C_ZERO, STORE_C_F32
	GEMM_BODY i8gemm_k_ld4_f, LOAD_C_ZERO, STORE_C_F32

	GEMM_BODY i8gemm_k_scaled_f, LOAD_C_ZERO, STORE_C_SCALED_F32, 2, 1
	GEMM_BODY i8gemm_k_scaled_b, LOAD_C_ZERO, STORE_C_SCALED_BF16, 1, 1

	GEMM_BODY i8gemm_k_ld_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
	GEMM_BODY i8gemm_k_ld1_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
	GEMM_BODY i8gemm_k_ld2_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
	GEMM_BODY i8gemm_k_ld4_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
