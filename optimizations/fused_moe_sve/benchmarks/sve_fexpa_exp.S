// SPDX-License-Identifier: Apache-2.0

.arch armv8.6-a+sve
.text

.equ VL_BYTES, FUSED_CPP_MOE_SVE_VECTOR_BITS / 8

.macro LOAD_PAIR gate, up
  ld1w z\gate\().s, p0/z, [x0]
  add x0, x0, #VL_BYTES
  ld1w z\up\().s, p0/z, [x0]
  add x0, x0, #VL_BYTES
.endm

.macro LOAD_POLY_CONSTANTS
  adrp x3, .Lpoly0
  add x3, x3, :lo12:.Lpoly0
  ld1w z6.s, p0/z, [x3]
  adrp x3, .Lpoly1
  add x3, x3, :lo12:.Lpoly1
  ld1w z7.s, p0/z, [x3]
  adrp x3, .Lln2
  add x3, x3, :lo12:.Lln2
  ld1rw z5.s, p0/z, [x3]
.endm

.macro LOAD_FEXPA_CONSTANTS
  adrp x3, .Lfexpa0
  add x3, x3, :lo12:.Lfexpa0
  ld1w z6.s, p0/z, [x3]
  adrp x3, .Lfexpa1
  add x3, x3, :lo12:.Lfexpa1
  ld1w z7.s, p0/z, [x3]
.endm

.macro LOAD_LIMIT
  adrp x3, .Lswiglu_limit
  add x3, x3, :lo12:.Lswiglu_limit
  ld1rw z4.s, p0/z, [x3]
.endm

.macro CLAMP gate, up, scope
  .ifc \scope,clamped_silu
    LOAD_LIMIT
    fmin z\gate\().s, p1/m, z\gate\().s, z4.s
    fmin z\up\().s, p1/m, z\up\().s, z4.s
    fneg z3.s, p1/m, z4.s
    fmax z\up\().s, p1/m, z\up\().s, z3.s
  .endif
.endm

.macro RANGE_POLY gate
  fneg z0.s, p1/m, z\gate\().s
  fmul z1.s, z0.s, z7.s[3]
  frintn z1.s, p1/m, z1.s
  fmls z0.s, p1/m, z1.s, z5.s
  fcvtzs z2.s, p1/m, z1.s
.endm

.macro POLY4 out
  dup z\out\().s, z6.s[2]
  dup z3.s, z7.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z7.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
.endm

.macro POLY5 out
  dup z\out\().s, z7.s[2]
  dup z3.s, z6.s[2]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z7.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z7.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
.endm

.macro POLY6 out
  dup z\out\().s, z6.s[3]
  dup z3.s, z7.s[2]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[2]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z7.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z7.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
.endm

.macro SCALE_POLY out
  mov z4.s, #127
  add z2.s, z2.s, z4.s
  lsl z2.s, z2.s, #23
  fmul z\out\().s, p1/m, z\out\().s, z2.s
.endm

.macro RANGE_FEXPA gate
  fneg z0.s, p1/m, z\gate\().s
  dup z1.s, z7.s[1]
  dup z3.s, z7.s[0]
  fmla z1.s, p1/m, z0.s, z3.s
  dup z3.s, z7.s[1]
  fsub z2.s, z1.s, z3.s
  dup z3.s, z7.s[2]
  fmls z0.s, p1/m, z2.s, z3.s
  dup z3.s, z6.s[2]
  fmls z0.s, p1/m, z2.s, z3.s
  fexpa z2.s, z1.s
.endm

.macro FEXPA_POLY2 out
  dup z\out\().s, z6.s[1]
  dup z3.s, z6.s[3]
  fmad z\out\().s, p1/m, z0.s, z3.s
  fmul z\out\().s, z\out\().s, z0.s
  fmad z\out\().s, p1/m, z2.s, z2.s
.endm

.macro FINISH gate, up, scope
  .ifc \scope,exp
    mov z\gate\().d, z\up\().d
  .else
    dup z4.s, z6.s[0]
    fadd z\up\().s, p1/m, z\up\().s, z4.s
    fdiv z\gate\().s, p1/m, z\gate\().s, z\up\().s
  .endif
.endm

.macro ROW_POLY gate, up, degree, scope
  CLAMP \gate, \up, \scope
  RANGE_POLY \gate
  .ifnc \scope,exp
    fmul z\gate\().s, p1/m, z\gate\().s, z\up\().s
  .endif
  POLY\degree \up
  SCALE_POLY \up
  FINISH \gate, \up, \scope
.endm

.macro ROW_FEXPA gate, up, scope
  CLAMP \gate, \up, \scope
  RANGE_FEXPA \gate
  .ifnc \scope,exp
    fmul z\gate\().s, p1/m, z\gate\().s, z\up\().s
  .endif
  FEXPA_POLY2 \up
  FINISH \gate, \up, \scope
.endm

.macro STORE reg
  st1w z\reg\().s, p0, [x1]
  add x1, x1, #VL_BYTES
.endm

.macro FUNCTION name, rows, variant, scope
  .global \name
  .type \name, %function
\name:
  .if \rows == 12
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
  .endif
  ptrue p0.s
  ptrue p1.s
  .if \rows == 8
    LOAD_PAIR 16, 17
    LOAD_PAIR 18, 19
    LOAD_PAIR 20, 21
    LOAD_PAIR 22, 23
    LOAD_PAIR 24, 25
    LOAD_PAIR 26, 27
    LOAD_PAIR 28, 29
    LOAD_PAIR 30, 31
  .else
    LOAD_PAIR 8, 9
    LOAD_PAIR 10, 11
    LOAD_PAIR 12, 13
    LOAD_PAIR 14, 15
    LOAD_PAIR 16, 17
    LOAD_PAIR 18, 19
    LOAD_PAIR 20, 21
    LOAD_PAIR 22, 23
    LOAD_PAIR 24, 25
    LOAD_PAIR 26, 27
    LOAD_PAIR 28, 29
    LOAD_PAIR 30, 31
  .endif
  .ifc \variant,fexpa
    LOAD_FEXPA_CONSTANTS
  .else
    LOAD_POLY_CONSTANTS
  .endif
  .if \rows == 8
    ROW_\variant 16, 17, \scope
    ROW_\variant 18, 19, \scope
    ROW_\variant 20, 21, \scope
    ROW_\variant 22, 23, \scope
    ROW_\variant 24, 25, \scope
    ROW_\variant 26, 27, \scope
    ROW_\variant 28, 29, \scope
    ROW_\variant 30, 31, \scope
  .else
    ROW_\variant 8, 9, \scope
    ROW_\variant 10, 11, \scope
    ROW_\variant 12, 13, \scope
    ROW_\variant 14, 15, \scope
    ROW_\variant 16, 17, \scope
    ROW_\variant 18, 19, \scope
    ROW_\variant 20, 21, \scope
    ROW_\variant 22, 23, \scope
    ROW_\variant 24, 25, \scope
    ROW_\variant 26, 27, \scope
    ROW_\variant 28, 29, \scope
    ROW_\variant 30, 31, \scope
  .endif
  .if \rows == 8
    STORE 16
    STORE 18
    STORE 20
    STORE 22
    STORE 24
    STORE 26
    STORE 28
    STORE 30
  .else
    STORE 8
    STORE 10
    STORE 12
    STORE 14
    STORE 16
    STORE 18
    STORE 20
    STORE 22
    STORE 24
    STORE 26
    STORE 28
    STORE 30
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
  .endif
  ret
  .size \name, .-\name
.endm

.macro INSTANTIATE rows, variant, scope
  FUNCTION sve_fexpa_m\rows\()_\variant\()_\scope, \rows, \variant, \scope
.endm

// Gas macro concatenation is simpler with explicit degree wrappers.
.macro ROW_poly4 gate, up, scope
  ROW_POLY \gate, \up, 4, \scope
.endm
.macro ROW_poly5 gate, up, scope
  ROW_POLY \gate, \up, 5, \scope
.endm
.macro ROW_poly6 gate, up, scope
  ROW_POLY \gate, \up, 6, \scope
.endm
.irp rows,8,12
  .irp variant,poly4,poly5,poly6,fexpa
    .irp scope,exp,silu,clamped_silu
      INSTANTIATE \rows, \variant, \scope
    .endr
  .endr
.endr

.section .rodata
.balign 32
.Lpoly0:
  .float 1.0, 0.5, 0.041666666666666664, 0.001388888888888889
  .float 1.0, 0.5, 0.041666666666666664, 0.001388888888888889
.Lpoly1:
  .float 1.0, 0.16666666666666666, 0.008333333333333333, 1.4426950408889634
  .float 1.0, 0.16666666666666666, 0.008333333333333333, 1.4426950408889634
.Lln2:
  .float 0.6931471805599453
.Lfexpa0:
  .float 1.0, 0.5000003576278687, 0.000001428606765330187, 1.000003695487976
  .float 1.0, 0.5000003576278687, 0.000001428606765330187, 1.000003695487976
.Lfexpa1:
  .float 1.4426950216293335, 196735.0, 0.693145751953125, 0.0
  .float 1.4426950216293335, 196735.0, 0.693145751953125, 0.0
.Lswiglu_limit:
  .float 10.0

.section .note.GNU-stack,"",%progbits
