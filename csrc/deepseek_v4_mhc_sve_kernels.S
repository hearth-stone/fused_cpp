// SPDX-License-Identifier: Apache-2.0

#if defined(__linux__) && defined(__aarch64__)

    .arch armv8.6-a+sve
    .text
    .p2align 4

    .macro SAVE_SIMD
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    .endm

    .macro RESTORE_SIMD
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    .endm

    .macro ZERO_ACCUMULATORS
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

    .macro M8_LOAD_ROW offset, c0, c1, c2
    add x15, x3, #\offset
    ld1w {z\c0\().s}, p0/z, [x15]
    ld1w {z\c1\().s}, p0/z, [x15, #1, mul vl]
    ld1w {z\c2\().s}, p0/z, [x15, #2, mul vl]
    .endm

    .macro M8_STORE_ROW offset, c0, c1, c2
    add x15, x3, #\offset
    st1w {z\c0\().s}, p0, [x15]
    st1w {z\c1\().s}, p0, [x15, #1, mul vl]
    st1w {z\c2\().s}, p0, [x15, #2, mul vl]
    .endm

    .macro M8_LOAD_A areg, ptr
    ldrh w14, [\ptr]
    lsl w14, w14, #16
    dup z\areg\().s, w14
    add \ptr, \ptr, #2
    .endm

    .macro M8_FMLA_ROW areg, c0, c1, c2, b0, b1, b2
    fmla z\c0\().s, p0/m, z\b0\().s, z\areg\().s
    fmla z\c1\().s, p0/m, z\b1\().s, z\areg\().s
    fmla z\c2\().s, p0/m, z\b2\().s, z\areg\().s
    .endm

    .macro M8_COMPUTE_BANK b0, b1, b2
    M8_LOAD_A 6, x0
    M8_LOAD_A 7, x6
    M8_FMLA_ROW 6, 8, 9, 10, \b0, \b1, \b2
    M8_LOAD_A 6, x7
    M8_FMLA_ROW 7, 11, 12, 13, \b0, \b1, \b2
    M8_LOAD_A 7, x8
    M8_FMLA_ROW 6, 14, 15, 16, \b0, \b1, \b2
    M8_LOAD_A 6, x9
    M8_FMLA_ROW 7, 17, 18, 19, \b0, \b1, \b2
    M8_LOAD_A 7, x10
    M8_FMLA_ROW 6, 20, 21, 22, \b0, \b1, \b2
    M8_LOAD_A 6, x11
    M8_FMLA_ROW 7, 23, 24, 25, \b0, \b1, \b2
    M8_LOAD_A 7, x12
    M8_FMLA_ROW 6, 26, 27, 28, \b0, \b1, \b2
    M8_FMLA_ROW 7, 29, 30, 31, \b0, \b1, \b2
    .endm

    // The rolling A pair hides LD1RH/LSL latency. The alternate B bank is
    // loaded after rows 0-2 while the current bank remains live.
    .macro M8_COMPUTE_CURRENT_LOAD_NEXT
    M8_LOAD_A 6, x0
    M8_LOAD_A 7, x6
    M8_FMLA_ROW 6, 8, 9, 10, 0, 1, 2
    ld1w {z3.s}, p0/z, [x13]
    M8_LOAD_A 6, x7
    M8_FMLA_ROW 7, 11, 12, 13, 0, 1, 2
    ld1w {z4.s}, p0/z, [x13, #1, mul vl]
    M8_LOAD_A 7, x8
    M8_FMLA_ROW 6, 14, 15, 16, 0, 1, 2
    ld1w {z5.s}, p0/z, [x13, #2, mul vl]
    add x13, x13, #96
    M8_LOAD_A 6, x9
    M8_FMLA_ROW 7, 17, 18, 19, 0, 1, 2
    M8_LOAD_A 7, x10
    M8_FMLA_ROW 6, 20, 21, 22, 0, 1, 2
    M8_LOAD_A 6, x11
    M8_FMLA_ROW 7, 23, 24, 25, 0, 1, 2
    M8_LOAD_A 7, x12
    M8_FMLA_ROW 6, 26, 27, 28, 0, 1, 2
    M8_FMLA_ROW 7, 29, 30, 31, 0, 1, 2
    .endm

    .macro M8_COMPUTE_NEXT_LOAD_CURRENT
    M8_LOAD_A 6, x0
    M8_LOAD_A 7, x6
    M8_FMLA_ROW 6, 8, 9, 10, 3, 4, 5
    ld1w {z0.s}, p0/z, [x13]
    M8_LOAD_A 6, x7
    M8_FMLA_ROW 7, 11, 12, 13, 3, 4, 5
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    M8_LOAD_A 7, x8
    M8_FMLA_ROW 6, 14, 15, 16, 3, 4, 5
    ld1w {z2.s}, p0/z, [x13, #2, mul vl]
    add x13, x13, #96
    M8_LOAD_A 6, x9
    M8_FMLA_ROW 7, 17, 18, 19, 3, 4, 5
    M8_LOAD_A 7, x10
    M8_FMLA_ROW 6, 20, 21, 22, 3, 4, 5
    M8_LOAD_A 6, x11
    M8_FMLA_ROW 7, 23, 24, 25, 3, 4, 5
    M8_LOAD_A 7, x12
    M8_FMLA_ROW 6, 26, 27, 28, 3, 4, 5
    M8_FMLA_ROW 7, 29, 30, 31, 3, 4, 5
    .endm

    .macro M7_ZERO_ACCUMULATORS
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

    .macro M7_LOAD_ROW offset, c0, c1, c2
    add x15, x3, #\offset
    ld1w {z\c0\().s}, p0/z, [x15]
    ld1w {z\c1\().s}, p0/z, [x15, #1, mul vl]
    ld1w {z\c2\().s}, p0/z, [x15, #2, mul vl]
    .endm

    .macro M7_STORE_ROW offset, c0, c1, c2
    add x15, x3, #\offset
    st1w {z\c0\().s}, p0, [x15]
    st1w {z\c1\().s}, p0, [x15, #1, mul vl]
    st1w {z\c2\().s}, p0, [x15, #2, mul vl]
    .endm

    .macro M7_LOAD_A_K4 areg, ptr
    ldr d\areg, [\ptr], #8
    shll v\areg\().4s, v\areg\().4h, #16
    mov z\areg\().q, q\areg
    .endm

    .macro M7_FMLA_COLUMN breg, lane, c0, c1, c2, c3, c4, c5, c6
    fmla z\c0\().s, z\breg\().s, z0.s[\lane]
    fmla z\c1\().s, z\breg\().s, z1.s[\lane]
    fmla z\c2\().s, z\breg\().s, z2.s[\lane]
    fmla z\c3\().s, z\breg\().s, z3.s[\lane]
    fmla z\c4\().s, z\breg\().s, z4.s[\lane]
    fmla z\c5\().s, z\breg\().s, z5.s[\lane]
    fmla z\c6\().s, z\breg\().s, z6.s[\lane]
    .endm

    .macro M7_COMPUTE_LANE_LOAD_NEXT lane
    M7_FMLA_COLUMN 7, \lane, 11,14,17,20,23,26,29
    ld1w {z7.s}, p0/z, [x12]
    M7_FMLA_COLUMN 8, \lane, 12,15,18,21,24,27,30
    ld1w {z8.s}, p0/z, [x12, #1, mul vl]
    M7_FMLA_COLUMN 9, \lane, 13,16,19,22,25,28,31
    ld1w {z9.s}, p0/z, [x12, #2, mul vl]
    add x12, x12, #96
    .endm

    .macro M7_COMPUTE_LANE lane
    M7_FMLA_COLUMN 7, \lane, 11,14,17,20,23,26,29
    M7_FMLA_COLUMN 8, \lane, 12,15,18,21,24,27,30
    M7_FMLA_COLUMN 9, \lane, 13,16,19,22,25,28,31
    .endm

    .macro M7_SCALAR_ROW ptr, c0, c1, c2
    ldrh w14, [\ptr], #2
    lsl w14, w14, #16
    dup z10.s, w14
    fmla z\c0\().s, p0/m, z7.s, z10.s
    fmla z\c1\().s, p0/m, z8.s, z10.s
    fmla z\c2\().s, p0/m, z9.s, z10.s
    .endm

    .global deepseek_v4_mhc_m7n24_sve
    .hidden deepseek_v4_mhc_m7n24_sve
    .type deepseek_v4_mhc_m7n24_sve, %function
deepseek_v4_mhc_m7n24_sve:
    SAVE_SIMD
    ptrue p0.s
    cbz w5, .Lm7_zero
    M7_LOAD_ROW 0, 11,12,13
    M7_LOAD_ROW 96, 14,15,16
    M7_LOAD_ROW 192, 17,18,19
    M7_LOAD_ROW 288, 20,21,22
    M7_LOAD_ROW 384, 23,24,25
    M7_LOAD_ROW 480, 26,27,28
    M7_LOAD_ROW 576, 29,30,31
    b .Lm7_init_done
.Lm7_zero:
    M7_ZERO_ACCUMULATORS
.Lm7_init_done:
    cbz x4, .Lm7_store
    lsl x16, x1, #1
    add x6, x0, x16
    add x7, x6, x16
    add x8, x7, x16
    add x9, x8, x16
    add x10, x9, x16
    add x11, x10, x16
    mov x12, x2
    cmp x4, #4
    b.lt .Lm7_k_tail
.Lm7_k4_loop:
    M7_LOAD_A_K4 0, x0
    M7_LOAD_A_K4 1, x6
    M7_LOAD_A_K4 2, x7
    M7_LOAD_A_K4 3, x8
    M7_LOAD_A_K4 4, x9
    M7_LOAD_A_K4 5, x10
    M7_LOAD_A_K4 6, x11
    ld1w {z7.s}, p0/z, [x12]
    ld1w {z8.s}, p0/z, [x12, #1, mul vl]
    ld1w {z9.s}, p0/z, [x12, #2, mul vl]
    add x12, x12, #96
    M7_COMPUTE_LANE_LOAD_NEXT 0
    M7_COMPUTE_LANE_LOAD_NEXT 1
    M7_COMPUTE_LANE_LOAD_NEXT 2
    M7_COMPUTE_LANE 3
    sub x4, x4, #4
    cmp x4, #4
    b.ge .Lm7_k4_loop
.Lm7_k_tail:
    cbz x4, .Lm7_store
.Lm7_k_tail_loop:
    ld1w {z7.s}, p0/z, [x12]
    ld1w {z8.s}, p0/z, [x12, #1, mul vl]
    ld1w {z9.s}, p0/z, [x12, #2, mul vl]
    add x12, x12, #96
    M7_SCALAR_ROW x0, 11,12,13
    M7_SCALAR_ROW x6, 14,15,16
    M7_SCALAR_ROW x7, 17,18,19
    M7_SCALAR_ROW x8, 20,21,22
    M7_SCALAR_ROW x9, 23,24,25
    M7_SCALAR_ROW x10, 26,27,28
    M7_SCALAR_ROW x11, 29,30,31
    subs x4, x4, #1
    b.ne .Lm7_k_tail_loop
.Lm7_store:
    M7_STORE_ROW 0, 11,12,13
    M7_STORE_ROW 96, 14,15,16
    M7_STORE_ROW 192, 17,18,19
    M7_STORE_ROW 288, 20,21,22
    M7_STORE_ROW 384, 23,24,25
    M7_STORE_ROW 480, 26,27,28
    M7_STORE_ROW 576, 29,30,31
    RESTORE_SIMD
    ret
    .size deepseek_v4_mhc_m7n24_sve, .-deepseek_v4_mhc_m7n24_sve

    .global deepseek_v4_mhc_m8n24_sve
    .hidden deepseek_v4_mhc_m8n24_sve
    .type deepseek_v4_mhc_m8n24_sve, %function
deepseek_v4_mhc_m8n24_sve:
    SAVE_SIMD
    ptrue p0.s
    cbz w5, .Lm8_zero
    M8_LOAD_ROW 0, 8, 9, 10
    M8_LOAD_ROW 96, 11, 12, 13
    M8_LOAD_ROW 192, 14, 15, 16
    M8_LOAD_ROW 288, 17, 18, 19
    M8_LOAD_ROW 384, 20, 21, 22
    M8_LOAD_ROW 480, 23, 24, 25
    M8_LOAD_ROW 576, 26, 27, 28
    M8_LOAD_ROW 672, 29, 30, 31
    b .Lm8_init_done
.Lm8_zero:
    ZERO_ACCUMULATORS
.Lm8_init_done:
    cbz x4, .Lm8_store
    lsl x16, x1, #1
    add x6, x0, x16
    add x7, x6, x16
    add x8, x7, x16
    add x9, x8, x16
    add x10, x9, x16
    add x11, x10, x16
    add x12, x11, x16
    mov x13, x2
    ld1w {z0.s}, p0/z, [x13]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    ld1w {z2.s}, p0/z, [x13, #2, mul vl]
    add x13, x13, #96
    subs x4, x4, #1
    b.eq .Lm8_tail_current
.Lm8_k_loop:
    M8_COMPUTE_CURRENT_LOAD_NEXT
    subs x4, x4, #1
    b.eq .Lm8_tail_next
    M8_COMPUTE_NEXT_LOAD_CURRENT
    subs x4, x4, #1
    b.gt .Lm8_k_loop
.Lm8_tail_current:
    M8_COMPUTE_BANK 0, 1, 2
    b .Lm8_store
.Lm8_tail_next:
    M8_COMPUTE_BANK 3, 4, 5
.Lm8_store:
    M8_STORE_ROW 0, 8, 9, 10
    M8_STORE_ROW 96, 11, 12, 13
    M8_STORE_ROW 192, 14, 15, 16
    M8_STORE_ROW 288, 17, 18, 19
    M8_STORE_ROW 384, 20, 21, 22
    M8_STORE_ROW 480, 23, 24, 25
    M8_STORE_ROW 576, 26, 27, 28
    M8_STORE_ROW 672, 29, 30, 31
    RESTORE_SIMD
    ret
    .size deepseek_v4_mhc_m8n24_sve, .-deepseek_v4_mhc_m8n24_sve

    .macro M4_LOAD_ROW offset, c0, c1, c2, c3, c4, c5
    add x15, x3, #\offset
    ld1w {z\c0\().s}, p0/z, [x15]
    ld1w {z\c1\().s}, p0/z, [x15, #1, mul vl]
    ld1w {z\c2\().s}, p0/z, [x15, #2, mul vl]
    ld1w {z\c3\().s}, p0/z, [x15, #3, mul vl]
    ld1w {z\c4\().s}, p0/z, [x15, #4, mul vl]
    ld1w {z\c5\().s}, p0/z, [x15, #5, mul vl]
    .endm

    .macro M4_STORE_ROW offset, c0, c1, c2, c3, c4, c5
    add x15, x3, #\offset
    st1w {z\c0\().s}, p0, [x15]
    st1w {z\c1\().s}, p0, [x15, #1, mul vl]
    st1w {z\c2\().s}, p0, [x15, #2, mul vl]
    st1w {z\c3\().s}, p0, [x15, #3, mul vl]
    st1w {z\c4\().s}, p0, [x15, #4, mul vl]
    st1w {z\c5\().s}, p0, [x15, #5, mul vl]
    .endm

    .macro M4_LOAD_A_PAIR ptr0, ptr1
    ldrh w14, [\ptr0]
    lsl w14, w14, #16
    dup z6.s, w14
    ldrh w14, [\ptr1]
    lsl w14, w14, #16
    dup z7.s, w14
    add \ptr0, \ptr0, #2
    add \ptr1, \ptr1, #2
    .endm

    .macro M4_COMPUTE_PAIR c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15
    fmla z\c00\().s, p0/m, z0.s, z6.s
    fmla z\c10\().s, p0/m, z0.s, z7.s
    fmla z\c01\().s, p0/m, z1.s, z6.s
    fmla z\c11\().s, p0/m, z1.s, z7.s
    fmla z\c02\().s, p0/m, z2.s, z6.s
    fmla z\c12\().s, p0/m, z2.s, z7.s
    fmla z\c03\().s, p0/m, z3.s, z6.s
    fmla z\c13\().s, p0/m, z3.s, z7.s
    fmla z\c04\().s, p0/m, z4.s, z6.s
    fmla z\c14\().s, p0/m, z4.s, z7.s
    fmla z\c05\().s, p0/m, z5.s, z6.s
    fmla z\c15\().s, p0/m, z5.s, z7.s
    .endm

    .macro M4_COMPUTE_CURRENT_LOAD_NEXT
    M4_LOAD_A_PAIR x0, x6
    M4_COMPUTE_PAIR 8,9,10,11,12,13, 14,15,16,17,18,19
    M4_LOAD_A_PAIR x7, x8
    fmla z20.s, p0/m, z0.s, z6.s
    fmla z26.s, p0/m, z0.s, z7.s
    ld1w {z0.s}, p0/z, [x13]
    fmla z21.s, p0/m, z1.s, z6.s
    fmla z27.s, p0/m, z1.s, z7.s
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    fmla z22.s, p0/m, z2.s, z6.s
    fmla z28.s, p0/m, z2.s, z7.s
    ld1w {z2.s}, p0/z, [x13, #2, mul vl]
    fmla z23.s, p0/m, z3.s, z6.s
    fmla z29.s, p0/m, z3.s, z7.s
    ld1w {z3.s}, p0/z, [x13, #3, mul vl]
    fmla z24.s, p0/m, z4.s, z6.s
    fmla z30.s, p0/m, z4.s, z7.s
    ld1w {z4.s}, p0/z, [x13, #4, mul vl]
    fmla z25.s, p0/m, z5.s, z6.s
    fmla z31.s, p0/m, z5.s, z7.s
    ld1w {z5.s}, p0/z, [x13, #5, mul vl]
    add x13, x13, #96
    .endm

    .macro M4_COMPUTE_CURRENT
    M4_LOAD_A_PAIR x0, x6
    M4_COMPUTE_PAIR 8,9,10,11,12,13, 14,15,16,17,18,19
    M4_LOAD_A_PAIR x7, x8
    M4_COMPUTE_PAIR 20,21,22,23,24,25, 26,27,28,29,30,31
    .endm

    .global deepseek_v4_mhc_m4n24_sve
    .hidden deepseek_v4_mhc_m4n24_sve
    .type deepseek_v4_mhc_m4n24_sve, %function
deepseek_v4_mhc_m4n24_sve:
    SAVE_SIMD
    ptrue p0.s
    cbz w5, .Lm4_zero
    M4_LOAD_ROW 0, 8,9,10,11,12,13
    M4_LOAD_ROW 96, 14,15,16,17,18,19
    M4_LOAD_ROW 192, 20,21,22,23,24,25
    M4_LOAD_ROW 288, 26,27,28,29,30,31
    b .Lm4_init_done
.Lm4_zero:
    ZERO_ACCUMULATORS
.Lm4_init_done:
    cbz x4, .Lm4_store
    lsl x16, x1, #1
    add x6, x0, x16
    add x7, x6, x16
    add x8, x7, x16
    mov x13, x2
    ld1w {z0.s}, p0/z, [x13]
    ld1w {z1.s}, p0/z, [x13, #1, mul vl]
    ld1w {z2.s}, p0/z, [x13, #2, mul vl]
    ld1w {z3.s}, p0/z, [x13, #3, mul vl]
    ld1w {z4.s}, p0/z, [x13, #4, mul vl]
    ld1w {z5.s}, p0/z, [x13, #5, mul vl]
    add x13, x13, #96
    subs x4, x4, #1
    b.eq .Lm4_tail
.Lm4_k_loop:
    M4_COMPUTE_CURRENT_LOAD_NEXT
    subs x4, x4, #1
    b.gt .Lm4_k_loop
.Lm4_tail:
    M4_COMPUTE_CURRENT
.Lm4_store:
    M4_STORE_ROW 0, 8,9,10,11,12,13
    M4_STORE_ROW 96, 14,15,16,17,18,19
    M4_STORE_ROW 192, 20,21,22,23,24,25
    M4_STORE_ROW 288, 26,27,28,29,30,31
    RESTORE_SIMD
    ret
    .size deepseek_v4_mhc_m4n24_sve, .-deepseek_v4_mhc_m4n24_sve

    .section .note.GNU-stack,"",%progbits

#endif
