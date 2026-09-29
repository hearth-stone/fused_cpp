// i8gemm_pack_a_neon.S -- NEON A packing helpers for i8 GEMM.
//
// The M8 pack layout matches the first-N strided path in i8gemm_k.S:
// for each K16 block, store rows 0..7 low 8 bytes, then rows 0..7 high 8 bytes.

	.text
	.align 4

// void i8_pack_A_neon_m8_asm(const int8_t *A, int8_t *P, int K_r, int lda);
// x0 = A row0, x1 = packed destination, w2 = K_r, w3 = lda in bytes.
#ifdef __APPLE__
	.global _i8_pack_A_neon_m8_asm
_i8_pack_A_neon_m8_asm:
#else
	.global i8_pack_A_neon_m8_asm
	.type i8_pack_A_neon_m8_asm, %function
i8_pack_A_neon_m8_asm:
#endif
	sxtw x3, w3
	lsl  x11, x3, #1
	lsl  x12, x3, #2
	add  x13, x11, x3
	add  x4, x0, x3
	add  x5, x0, x11
	add  x6, x0, x13
	add  x7, x0, x12
	add  x8, x4, x12
	add  x9, x5, x12
	add  x10, x6, x12
	lsr  w11, w2, #4
	cbz  w11, .Li8_pack_A_neon_m8_done

.Li8_pack_A_neon_m8_loop:
	ldr  d0, [x0], #8
	ld1  {v0.d}[1], [x4], #8
	ldr  d1, [x5], #8
	ld1  {v1.d}[1], [x6], #8
	ldr  d2, [x7], #8
	ld1  {v2.d}[1], [x8], #8
	ldr  d3, [x9], #8
	ld1  {v3.d}[1], [x10], #8
	stp  q0, q1, [x1], #32
	stp  q2, q3, [x1], #32
	ldr  d0, [x0], #8
	ld1  {v0.d}[1], [x4], #8
	ldr  d1, [x5], #8
	ld1  {v1.d}[1], [x6], #8
	ldr  d2, [x7], #8
	ld1  {v2.d}[1], [x8], #8
	ldr  d3, [x9], #8
	ld1  {v3.d}[1], [x10], #8
	stp  q0, q1, [x1], #32
	stp  q2, q3, [x1], #32
	subs w11, w11, #1
	b.ne .Li8_pack_A_neon_m8_loop

.Li8_pack_A_neon_m8_done:
	ret

#ifndef __APPLE__
	.size i8_pack_A_neon_m8_asm, .-i8_pack_A_neon_m8_asm
#endif

// void i8_pack_A_neon_m4_asm(const int8_t *A, int8_t *P, int K_r, int lda);
#ifdef __APPLE__
	.global _i8_pack_A_neon_m4_asm
_i8_pack_A_neon_m4_asm:
#else
	.global i8_pack_A_neon_m4_asm
	.type i8_pack_A_neon_m4_asm, %function
i8_pack_A_neon_m4_asm:
#endif
	sxtw x3, w3
	lsl  x11, x3, #1
	add  x13, x11, x3
	add  x4, x0, x3
	add  x5, x0, x11
	add  x6, x0, x13
	lsr  w11, w2, #4
	cbz  w11, .Li8_pack_A_neon_m4_done

.Li8_pack_A_neon_m4_loop:
	ldr  d0, [x0], #8
	ld1  {v0.d}[1], [x4], #8
	ldr  d1, [x5], #8
	ld1  {v1.d}[1], [x6], #8
	stp  q0, q1, [x1], #32
	ldr  d0, [x0], #8
	ld1  {v0.d}[1], [x4], #8
	ldr  d1, [x5], #8
	ld1  {v1.d}[1], [x6], #8
	stp  q0, q1, [x1], #32
	subs w11, w11, #1
	b.ne .Li8_pack_A_neon_m4_loop

.Li8_pack_A_neon_m4_done:
	ret

#ifndef __APPLE__
	.size i8_pack_A_neon_m4_asm, .-i8_pack_A_neon_m4_asm
#endif

// void i8_pack_A_neon_m2_asm(const int8_t *A, int8_t *P, int K_r, int lda);
#ifdef __APPLE__
	.global _i8_pack_A_neon_m2_asm
_i8_pack_A_neon_m2_asm:
#else
	.global i8_pack_A_neon_m2_asm
	.type i8_pack_A_neon_m2_asm, %function
i8_pack_A_neon_m2_asm:
#endif
	sxtw x3, w3
	add  x4, x0, x3
	lsr  w11, w2, #4
	cbz  w11, .Li8_pack_A_neon_m2_done

.Li8_pack_A_neon_m2_loop:
	ldr  d0, [x0], #8
	ld1  {v0.d}[1], [x4], #8
	str  q0, [x1], #16
	ldr  d0, [x0], #8
	ld1  {v0.d}[1], [x4], #8
	str  q0, [x1], #16
	subs w11, w11, #1
	b.ne .Li8_pack_A_neon_m2_loop

.Li8_pack_A_neon_m2_done:
	ret

#ifndef __APPLE__
	.size i8_pack_A_neon_m2_asm, .-i8_pack_A_neon_m2_asm
#endif

// void i8_pack_A_neon_m1_asm(const int8_t *A, int8_t *P, int K_r, int lda);
#ifdef __APPLE__
	.global _i8_pack_A_neon_m1_asm
_i8_pack_A_neon_m1_asm:
#else
	.global i8_pack_A_neon_m1_asm
	.type i8_pack_A_neon_m1_asm, %function
i8_pack_A_neon_m1_asm:
#endif
	lsr  w11, w2, #4
	cbz  w11, .Li8_pack_A_neon_m1_done

.Li8_pack_A_neon_m1_loop:
	ldr  d0, [x0], #8
	mov  v0.d[1], xzr
	str  q0, [x1], #16
	ldr  d0, [x0], #8
	mov  v0.d[1], xzr
	str  q0, [x1], #16
	subs w11, w11, #1
	b.ne .Li8_pack_A_neon_m1_loop

.Li8_pack_A_neon_m1_done:
	ret

#ifndef __APPLE__
	.size i8_pack_A_neon_m1_asm, .-i8_pack_A_neon_m1_asm
#endif
