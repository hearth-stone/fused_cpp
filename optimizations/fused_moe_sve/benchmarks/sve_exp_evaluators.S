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

.macro RANGE gate
  fneg z0.s, p1/m, z\gate\().s
  fmul z1.s, z0.s, z7.s[2]
  frintn z1.s, p1/m, z1.s
  fmls z0.s, z1.s, z7.s[3]
  fcvtzs z2.s, p1/m, z1.s
.endm

.macro POLY_HORNER out
  dup z\out\().s, z7.s[1]
  dup z3.s, z7.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[3]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[2]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[1]
  fmad z\out\().s, p1/m, z0.s, z3.s
  dup z3.s, z6.s[0]
  fmad z\out\().s, p1/m, z0.s, z3.s
.endm

.macro POLY_EVEN_ODD out
  fmul z3.s, z0.s, z0.s
  fmul z4.s, z3.s, z3.s
  dup z\out\().s, z6.s[0]
  fmla z\out\().s, z3.s, z6.s[2]
  fmla z\out\().s, z4.s, z7.s[0]
  dup z5.s, z6.s[1]
  fmla z5.s, z3.s, z6.s[3]
  fmla z5.s, z4.s, z7.s[1]
  fmla z\out\().s, p1/m, z0.s, z5.s
.endm

.macro POLY_ESTRIN out
  fmul z3.s, z0.s, z0.s
  dup z\out\().s, z7.s[0]
  fmla z\out\().s, z0.s, z7.s[1]
  dup z4.s, z6.s[2]
  fmla z4.s, z0.s, z6.s[3]
  dup z5.s, z6.s[0]
  fmla z5.s, z0.s, z6.s[1]
  fmla z4.s, p1/m, z3.s, z\out\().s
  fmla z5.s, p1/m, z3.s, z4.s
  mov z\out\().d, z5.d
.endm

.macro ROW gate, up, evaluator, scope
  .ifc \scope,clamped_silu
    LOAD_LIMIT
    fmin z\gate\().s, p1/m, z\gate\().s, z5.s
    fmin z\up\().s, p1/m, z\up\().s, z5.s
    fneg z4.s, p1/m, z5.s
    fmax z\up\().s, p1/m, z\up\().s, z4.s
  .endif
  RANGE \gate
  .ifnc \scope,exp
    fmul z\gate\().s, p1/m, z\gate\().s, z\up\().s
  .endif
  .ifc \evaluator,horner
    POLY_HORNER \up
  .endif
  .ifc \evaluator,even_odd
    POLY_EVEN_ODD \up
  .endif
  .ifc \evaluator,estrin
    POLY_ESTRIN \up
  .endif
  mov z4.s, #127
  add z2.s, z2.s, z4.s
  lsl z2.s, z2.s, #23
  fmul z\up\().s, p1/m, z\up\().s, z2.s
  .ifc \scope,exp
    mov z\gate\().d, z\up\().d
  .else
    dup z4.s, z6.s[0]
    fadd z\up\().s, p1/m, z\up\().s, z4.s
    fdiv z\gate\().s, p1/m, z\gate\().s, z\up\().s
  .endif
.endm

.macro COMMON_LOADS
  adrp x3, .Lcoefficients0
  add x3, x3, :lo12:.Lcoefficients0
  ld1w z6.s, p0/z, [x3]
  adrp x3, .Lcoefficients1
  add x3, x3, :lo12:.Lcoefficients1
  ld1w z7.s, p0/z, [x3]
.endm

.macro LOAD_LIMIT
  adrp x3, .Lswiglu_limit
  add x3, x3, :lo12:.Lswiglu_limit
  ld1rw z5.s, p0/z, [x3]
.endm

.macro STORE reg
  st1w z\reg\().s, p0, [x1]
  add x1, x1, #VL_BYTES
.endm

.macro FUNCTION8 name, evaluator, scope
  .global \name
  .type \name, %function
\name:
  ptrue p0.s
  ptrue p1.s
  LOAD_PAIR 16, 17
  LOAD_PAIR 18, 19
  LOAD_PAIR 20, 21
  LOAD_PAIR 22, 23
  LOAD_PAIR 24, 25
  LOAD_PAIR 26, 27
  LOAD_PAIR 28, 29
  LOAD_PAIR 30, 31
  COMMON_LOADS
  ROW 16, 17, \evaluator, \scope
  ROW 18, 19, \evaluator, \scope
  ROW 20, 21, \evaluator, \scope
  ROW 22, 23, \evaluator, \scope
  ROW 24, 25, \evaluator, \scope
  ROW 26, 27, \evaluator, \scope
  ROW 28, 29, \evaluator, \scope
  ROW 30, 31, \evaluator, \scope
  STORE 16
  STORE 18
  STORE 20
  STORE 22
  STORE 24
  STORE 26
  STORE 28
  STORE 30
  ret
  .size \name, .-\name
.endm

.macro FUNCTION12 name, evaluator, scope
  .global \name
  .type \name, %function
\name:
  stp d8, d9, [sp, #-16]!
  stp d10, d11, [sp, #-16]!
  stp d12, d13, [sp, #-16]!
  stp d14, d15, [sp, #-16]!
  ptrue p0.s
  ptrue p1.s
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
  COMMON_LOADS
  ROW 8, 9, \evaluator, \scope
  ROW 10, 11, \evaluator, \scope
  ROW 12, 13, \evaluator, \scope
  ROW 14, 15, \evaluator, \scope
  ROW 16, 17, \evaluator, \scope
  ROW 18, 19, \evaluator, \scope
  ROW 20, 21, \evaluator, \scope
  ROW 22, 23, \evaluator, \scope
  ROW 24, 25, \evaluator, \scope
  ROW 26, 27, \evaluator, \scope
  ROW 28, 29, \evaluator, \scope
  ROW 30, 31, \evaluator, \scope
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
  ret
  .size \name, .-\name
.endm

.macro INSTANTIATE rows, evaluator, scope
  FUNCTION\rows sve_exp_m\rows\()_\evaluator\()_\scope, \evaluator, \scope
.endm

.irp rows,8,12
  .irp evaluator,horner,even_odd,estrin
    INSTANTIATE \rows, \evaluator, exp
    INSTANTIATE \rows, \evaluator, silu
    INSTANTIATE \rows, \evaluator, clamped_silu
  .endr
.endr

.section .rodata
.balign 32
.Lcoefficients0:
  .float 1.0, 1.0, 0.5, 0.16666666, 1.0, 1.0, 0.5, 0.16666666
.Lcoefficients1:
  .float 0.04166666, 0.00833333, 1.4426950408889634, 0.6931471805599453
  .float 0.04166666, 0.00833333, 1.4426950408889634, 0.6931471805599453
.Lswiglu_limit:
  .float 10.0

.section .note.GNU-stack,"",%progbits
