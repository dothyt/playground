	.text
	.file	"memcpy_test.cpp"
	.globl	_Z13avx512_memcpyPvS_S_m        # -- Begin function _Z13avx512_memcpyPvS_S_m
	.p2align	4, 0x90
	.type	_Z13avx512_memcpyPvS_S_m,@function
_Z13avx512_memcpyPvS_S_m:               # @_Z13avx512_memcpyPvS_S_m
	.cfi_startproc
# %bb.0:
	cmpq	$513, %rcx                      # imm = 0x201
	jb	.LBB0_3
# %bb.1:
	movl	$512, %eax                      # imm = 0x200
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	prefetcht1	-512(%rdx,%rax)
	prefetcht1	-448(%rdx,%rax)
	prefetcht1	-384(%rdx,%rax)
	prefetcht1	-320(%rdx,%rax)
	prefetcht1	-256(%rdx,%rax)
	prefetcht1	-192(%rdx,%rax)
	prefetcht1	-128(%rdx,%rax)
	prefetcht1	-64(%rdx,%rax)
	vmovaps	-512(%rsi,%rax), %zmm0
	vmovaps	-448(%rsi,%rax), %zmm1
	vmovaps	-384(%rsi,%rax), %zmm2
	vmovaps	-320(%rsi,%rax), %zmm3
	vmovaps	-256(%rsi,%rax), %zmm4
	vmovaps	-192(%rsi,%rax), %zmm5
	vmovaps	-128(%rsi,%rax), %zmm6
	vmovaps	-64(%rsi,%rax), %zmm7
	vmovaps	-512(%rdx,%rax), %zmm8
	vmovaps	-448(%rdx,%rax), %zmm9
	vmovaps	-384(%rdx,%rax), %zmm10
	vmovaps	-320(%rdx,%rax), %zmm11
	vmovaps	-256(%rdx,%rax), %zmm12
	vmovaps	-192(%rdx,%rax), %zmm13
	vmovaps	-128(%rdx,%rax), %zmm14
	vmovaps	-64(%rdx,%rax), %zmm15
	prefetcht1	-512(%rdi,%rax)
	prefetcht1	-448(%rdi,%rax)
	prefetcht1	-384(%rdi,%rax)
	prefetcht1	-320(%rdi,%rax)
	prefetcht1	-256(%rdi,%rax)
	prefetcht1	-192(%rdi,%rax)
	prefetcht1	-128(%rdi,%rax)
	prefetcht1	-64(%rdi,%rax)
	vmovaps	%zmm0, -512(%rdi,%rax)
	vmovaps	%zmm1, -448(%rdi,%rax)
	vmovaps	%zmm2, -384(%rdi,%rax)
	vmovaps	%zmm3, -320(%rdi,%rax)
	vmovaps	%zmm4, -256(%rdi,%rax)
	vmovaps	%zmm5, -192(%rdi,%rax)
	vmovaps	%zmm6, -128(%rdi,%rax)
	vmovaps	%zmm7, -64(%rdi,%rax)
	vmovaps	%zmm8, -512(%rsi,%rax)
	vmovaps	%zmm9, -448(%rsi,%rax)
	vmovaps	%zmm10, -384(%rsi,%rax)
	vmovaps	%zmm11, -320(%rsi,%rax)
	vmovaps	%zmm12, -256(%rsi,%rax)
	vmovaps	%zmm13, -192(%rsi,%rax)
	vmovaps	%zmm14, -128(%rsi,%rax)
	vmovaps	%zmm15, -64(%rsi,%rax)
	addq	$512, %rax                      # imm = 0x200
	cmpq	%rcx, %rax
	jb	.LBB0_2
.LBB0_3:
	vzeroupper
	retq
.Lfunc_end0:
	.size	_Z13avx512_memcpyPvS_S_m, .Lfunc_end0-_Z13avx512_memcpyPvS_S_m
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$64, %edi
	movl	$4096000000, %esi               # imm = 0xF4240000
	callq	aligned_alloc@PLT
	movq	%rax, %r12
	movl	$64, %edi
	movl	$4096000000, %esi               # imm = 0xF4240000
	callq	aligned_alloc@PLT
	movq	%rax, %rbx
	movl	$64, %edi
	movl	$4096000000, %esi               # imm = 0xF4240000
	callq	aligned_alloc@PLT
	addq	$448, %rbx                      # imm = 0x1C0
	movq	%rbx, 32(%rsp)                  # 8-byte Spill
	addq	$448, %rax                      # imm = 0x1C0
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	addq	$448, %r12                      # imm = 0x1C0
	xorl	%r15d, %r15d
	xorl	%r13d, %r13d
	xorl	%ebp, %ebp
	xorl	%eax, %eax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	%r12, 16(%rsp)                  # 8-byte Spill
	jmp	.LBB1_1
	.p2align	4, 0x90
.LBB1_6:                                #   in Loop: Header=BB1_1 Depth=1
	movq	%r14, (%rbp)
	addq	$8, %rbp
	incl	%r15d
	cmpl	$20, %r15d
	je	.LBB1_8
.LBB1_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_2 Depth 2
                                        #       Child Loop BB1_3 Depth 3
	callq	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, %rbx
	movq	%r12, %rax
	movq	24(%rsp), %rcx                  # 8-byte Reload
	movq	32(%rsp), %rdx                  # 8-byte Reload
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB1_2:                                #   Parent Loop BB1_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_3 Depth 3
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB1_3:                                #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_2 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	prefetcht1	-448(%rax,%rdi)
	prefetcht1	-384(%rax,%rdi)
	prefetcht1	-320(%rax,%rdi)
	prefetcht1	-256(%rax,%rdi)
	prefetcht1	-192(%rax,%rdi)
	prefetcht1	-128(%rax,%rdi)
	prefetcht1	-64(%rax,%rdi)
	prefetcht1	(%rax,%rdi)
	vmovaps	-448(%rcx,%rdi), %zmm0
	vmovaps	-384(%rcx,%rdi), %zmm1
	vmovaps	-320(%rcx,%rdi), %zmm2
	vmovaps	-256(%rcx,%rdi), %zmm3
	vmovaps	-192(%rcx,%rdi), %zmm4
	vmovaps	-128(%rcx,%rdi), %zmm5
	vmovaps	-64(%rcx,%rdi), %zmm6
	vmovaps	(%rcx,%rdi), %zmm7
	vmovaps	-448(%rax,%rdi), %zmm8
	vmovaps	-384(%rax,%rdi), %zmm9
	vmovaps	-320(%rax,%rdi), %zmm10
	vmovaps	-256(%rax,%rdi), %zmm11
	vmovaps	-192(%rax,%rdi), %zmm12
	vmovaps	-128(%rax,%rdi), %zmm13
	vmovaps	-64(%rax,%rdi), %zmm14
	vmovaps	(%rax,%rdi), %zmm15
	prefetcht1	-448(%rdx,%rdi)
	prefetcht1	-384(%rdx,%rdi)
	prefetcht1	-320(%rdx,%rdi)
	prefetcht1	-256(%rdx,%rdi)
	prefetcht1	-192(%rdx,%rdi)
	prefetcht1	-128(%rdx,%rdi)
	prefetcht1	-64(%rdx,%rdi)
	prefetcht1	(%rdx,%rdi)
	vmovaps	%zmm0, -448(%rdx,%rdi)
	vmovaps	%zmm1, -384(%rdx,%rdi)
	vmovaps	%zmm2, -320(%rdx,%rdi)
	vmovaps	%zmm3, -256(%rdx,%rdi)
	vmovaps	%zmm4, -192(%rdx,%rdi)
	vmovaps	%zmm5, -128(%rdx,%rdi)
	vmovaps	%zmm6, -64(%rdx,%rdi)
	vmovaps	%zmm7, (%rdx,%rdi)
	vmovaps	%zmm8, -448(%rcx,%rdi)
	vmovaps	%zmm9, -384(%rcx,%rdi)
	vmovaps	%zmm10, -320(%rcx,%rdi)
	vmovaps	%zmm11, -256(%rcx,%rdi)
	vmovaps	%zmm12, -192(%rcx,%rdi)
	vmovaps	%zmm13, -128(%rcx,%rdi)
	vmovaps	%zmm14, -64(%rcx,%rdi)
	vmovaps	%zmm15, (%rcx,%rdi)
	addq	$512, %rdi                      # imm = 0x200
	cmpq	$3584, %rdi                     # imm = 0xE00
	jb	.LBB1_3
# %bb.4:                                #   in Loop: Header=BB1_2 Depth=2
	incq	%rsi
	addq	$4096, %rdx                     # imm = 0x1000
	addq	$4096, %rcx                     # imm = 0x1000
	addq	$4096, %rax                     # imm = 0x1000
	cmpq	$1000000, %rsi                  # imm = 0xF4240
	jne	.LBB1_2
# %bb.5:                                #   in Loop: Header=BB1_1 Depth=1
	vzeroupper
	callq	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, %r14
	subq	%rbx, %r14
	cmpq	%r13, %rbp
	jne	.LBB1_6
# %bb.17:                               #   in Loop: Header=BB1_1 Depth=1
	subq	8(%rsp), %r13                   # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r13
	je	.LBB1_18
# %bb.20:                               #   in Loop: Header=BB1_1 Depth=1
	movq	%r13, %rbp
	sarq	$3, %rbp
	cmpq	$1, %rbp
	movq	%rbp, %rax
	adcq	$0, %rax
	leaq	(%rax,%rbp), %r12
	movabsq	$1152921504606846975, %rcx      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rcx, %r12
	cmovaeq	%rcx, %r12
	addq	%rbp, %rax
	cmovbq	%rcx, %r12
	testq	%r12, %r12
	je	.LBB1_21
# %bb.22:                               #   in Loop: Header=BB1_1 Depth=1
	leaq	(,%r12,8), %rdi
.Ltmp0:
	callq	_Znwm@PLT
.Ltmp1:
# %bb.23:                               #   in Loop: Header=BB1_1 Depth=1
	movq	%rax, %rbx
	movq	%r14, (%rbx,%rbp,8)
	testq	%r13, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	jle	.LBB1_26
.LBB1_25:                               #   in Loop: Header=BB1_1 Depth=1
	movq	%rbx, %rdi
	movq	%r14, %rsi
	movq	%r13, %rdx
	callq	memmove@PLT
.LBB1_26:                               #   in Loop: Header=BB1_1 Depth=1
	testq	%r14, %r14
	je	.LBB1_28
# %bb.27:                               #   in Loop: Header=BB1_1 Depth=1
	movq	%r14, %rdi
	callq	_ZdlPv@PLT
.LBB1_28:                               #   in Loop: Header=BB1_1 Depth=1
	addq	%rbx, %r13
	movq	%r13, %rbp
	leaq	(%rbx,%r12,8), %r13
	movq	%rbx, 8(%rsp)                   # 8-byte Spill
	movq	16(%rsp), %r12                  # 8-byte Reload
	addq	$8, %rbp
	incl	%r15d
	cmpl	$20, %r15d
	jne	.LBB1_1
	jmp	.LBB1_8
.LBB1_21:                               #   in Loop: Header=BB1_1 Depth=1
	xorl	%ebx, %ebx
	movq	%r14, (%rbx,%rbp,8)
	testq	%r13, %r13
	movq	8(%rsp), %r14                   # 8-byte Reload
	jg	.LBB1_25
	jmp	.LBB1_26
.LBB1_8:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	cmpq	%rbp, %rdi
	je	.LBB1_9
# %bb.10:
	movq	%rbp, %rbx
	subq	%rdi, %rbx
	sarq	$3, %rbx
	lzcntq	%rbx, %rdx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp3:
	movq	%rbp, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp4:
# %bb.11:
.Ltmp5:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	movq	%rbp, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp6:
	jmp	.LBB1_12
.LBB1_9:
	xorl	%ebx, %ebx
.LBB1_12:
	andq	$-2, %rbx
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	(%rax,%rbx,4), %r14
	movabsq	$1000000000000000, %rax         # imm = 0x38D7EA4C68000
	xorl	%edx, %edx
	divq	%r14
.Ltmp7:
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movq	%rax, %rsi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp8:
# %bb.13:
.Ltmp9:
	movq	%rax, %rbx
	leaq	.L.str(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp10:
# %bb.14:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	movq	240(%rbx,%rax), %r15
	testq	%r15, %r15
	je	.LBB1_38
# %bb.15:
	cmpb	$0, 56(%r15)
	je	.LBB1_31
# %bb.16:
	movzbl	67(%r15), %eax
	jmp	.LBB1_33
.LBB1_31:
.Ltmp11:
	movq	%r15, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
.Ltmp12:
# %bb.32:
	movq	(%r15), %rax
.Ltmp13:
	movq	%r15, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp14:
.LBB1_33:
.Ltmp15:
	movsbl	%al, %esi
	movq	%rbx, %rdi
	callq	_ZNSo3putEc@PLT
.Ltmp16:
# %bb.34:
.Ltmp17:
	movq	%rax, %rdi
	callq	_ZNSo5flushEv@PLT
.Ltmp18:
# %bb.35:
	movabsq	$4835703278458516699, %rax      # imm = 0x431BDE82D7B634DB
	movq	%r14, %rdx
	mulxq	%rax, %rsi, %rsi
	shrq	$18, %rsi
.Ltmp19:
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	callq	_ZNSo9_M_insertImEERSoT_@PLT
.Ltmp20:
# %bb.36:
.Ltmp21:
	movq	%rax, %rbx
	leaq	.L.str.1(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp22:
# %bb.37:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	movq	240(%rbx,%rax), %r14
	testq	%r14, %r14
	je	.LBB1_38
# %bb.40:
	cmpb	$0, 56(%r14)
	je	.LBB1_42
# %bb.41:
	movzbl	67(%r14), %eax
	jmp	.LBB1_44
.LBB1_42:
.Ltmp23:
	movq	%r14, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
.Ltmp24:
# %bb.43:
	movq	(%r14), %rax
.Ltmp25:
	movq	%r14, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp26:
.LBB1_44:
.Ltmp27:
	movsbl	%al, %esi
	movq	%rbx, %rdi
	callq	_ZNSo3putEc@PLT
.Ltmp28:
# %bb.45:
.Ltmp29:
	movq	%rax, %rdi
	callq	_ZNSo5flushEv@PLT
.Ltmp30:
# %bb.46:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	callq	_ZdlPv@PLT
	xorl	%eax, %eax
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB1_18:
	.cfi_def_cfa_offset 96
.Ltmp34:
	leaq	.L.str.2(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp35:
# %bb.19:
.LBB1_38:
.Ltmp31:
	callq	_ZSt16__throw_bad_castv@PLT
.Ltmp32:
# %bb.39:
.LBB1_29:
.Ltmp2:
	jmp	.LBB1_48
.LBB1_30:
.Ltmp36:
	jmp	.LBB1_48
.LBB1_47:
.Ltmp33:
.LBB1_48:
	movq	%rax, %rbx
	cmpq	$0, 8(%rsp)                     # 8-byte Folded Reload
	je	.LBB1_50
# %bb.49:
	movq	8(%rsp), %rdi                   # 8-byte Reload
	callq	_ZdlPv@PLT
.LBB1_50:
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table1:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 1 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp3-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp30-.Ltmp3                 #   Call between .Ltmp3 and .Ltmp30
	.uleb128 .Ltmp33-.Lfunc_begin0          #     jumps to .Ltmp33
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp34-.Lfunc_begin0          # >> Call Site 4 <<
	.uleb128 .Ltmp35-.Ltmp34                #   Call between .Ltmp34 and .Ltmp35
	.uleb128 .Ltmp36-.Lfunc_begin0          #     jumps to .Ltmp36
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp31-.Lfunc_begin0          # >> Call Site 5 <<
	.uleb128 .Ltmp32-.Ltmp31                #   Call between .Ltmp31 and .Ltmp32
	.uleb128 .Ltmp33-.Lfunc_begin0          #     jumps to .Ltmp33
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp32-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Lfunc_end1-.Ltmp32            #   Call between .Ltmp32 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.p2align	4, 0x90
	.type	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,@function
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_: # @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbp
	subq	%rdi, %rbp
	sarq	$3, %rbp
	cmpq	$17, %rbp
	jl	.LBB2_38
# %bb.1:
	movq	%rdx, %r14
	movq	%rdi, %rbx
	leaq	8(%rdi), %r12
	movq	$-8, %r13
	subq	%rdi, %r13
	jmp	.LBB2_2
	.p2align	4, 0x90
.LBB2_37:                               #   in Loop: Header=BB2_2 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	sarq	$3, %rbp
	movq	%r15, %rsi
	cmpq	$16, %rbp
	jle	.LBB2_38
.LBB2_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_31 Depth 2
                                        #       Child Loop BB2_32 Depth 3
                                        #       Child Loop BB2_34 Depth 3
	subq	$1, %r14
	jb	.LBB2_3
# %bb.19:                               #   in Loop: Header=BB2_2 Depth=1
	shrq	%rbp
	movq	8(%rbx), %rcx
	movq	(%rbx,%rbp,8), %rdx
	movq	-8(%rsi), %rax
	cmpq	%rdx, %rcx
	jae	.LBB2_25
# %bb.20:                               #   in Loop: Header=BB2_2 Depth=1
	cmpq	%rax, %rdx
	jae	.LBB2_22
# %bb.21:                               #   in Loop: Header=BB2_2 Depth=1
	movq	(%rbx), %rax
	movq	%rdx, (%rbx)
	movq	%rax, (%rbx,%rbp,8)
	jmp	.LBB2_30
	.p2align	4, 0x90
.LBB2_25:                               #   in Loop: Header=BB2_2 Depth=1
	cmpq	%rax, %rcx
	jae	.LBB2_27
# %bb.26:                               #   in Loop: Header=BB2_2 Depth=1
	movq	(%rbx), %rax
	movq	%rcx, (%rbx)
	movq	%rax, 8(%rbx)
	jmp	.LBB2_30
	.p2align	4, 0x90
.LBB2_22:                               #   in Loop: Header=BB2_2 Depth=1
	movq	(%rbx), %rdx
	cmpq	%rax, %rcx
	jae	.LBB2_24
# %bb.23:                               #   in Loop: Header=BB2_2 Depth=1
	movq	%rax, (%rbx)
	movq	%rdx, -8(%rsi)
	jmp	.LBB2_30
	.p2align	4, 0x90
.LBB2_27:                               #   in Loop: Header=BB2_2 Depth=1
	movq	(%rbx), %rcx
	cmpq	%rax, %rdx
	jae	.LBB2_29
# %bb.28:                               #   in Loop: Header=BB2_2 Depth=1
	movq	%rax, (%rbx)
	movq	%rcx, -8(%rsi)
	jmp	.LBB2_30
.LBB2_24:                               #   in Loop: Header=BB2_2 Depth=1
	movq	%rcx, (%rbx)
	movq	%rdx, 8(%rbx)
	jmp	.LBB2_30
.LBB2_29:                               #   in Loop: Header=BB2_2 Depth=1
	movq	%rdx, (%rbx)
	movq	%rcx, (%rbx,%rbp,8)
	.p2align	4, 0x90
.LBB2_30:                               #   in Loop: Header=BB2_2 Depth=1
	movq	%r12, %rax
	movq	%rsi, %rcx
	.p2align	4, 0x90
.LBB2_31:                               #   Parent Loop BB2_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_32 Depth 3
                                        #       Child Loop BB2_34 Depth 3
	movq	(%rbx), %rdx
	leaq	(%rax,%r13), %rbp
	.p2align	4, 0x90
.LBB2_32:                               #   Parent Loop BB2_2 Depth=1
                                        #     Parent Loop BB2_31 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	(%rax), %rdi
	addq	$8, %rax
	addq	$8, %rbp
	cmpq	%rdx, %rdi
	jb	.LBB2_32
# %bb.33:                               #   in Loop: Header=BB2_31 Depth=2
	leaq	-8(%rax), %r15
	.p2align	4, 0x90
.LBB2_34:                               #   Parent Loop BB2_2 Depth=1
                                        #     Parent Loop BB2_31 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	-8(%rcx), %r8
	addq	$-8, %rcx
	cmpq	%r8, %rdx
	jb	.LBB2_34
# %bb.35:                               #   in Loop: Header=BB2_31 Depth=2
	cmpq	%rcx, %r15
	jae	.LBB2_37
# %bb.36:                               #   in Loop: Header=BB2_31 Depth=2
	movq	%r8, (%r15)
	movq	%rdi, (%rcx)
	jmp	.LBB2_31
.LBB2_3:
	leaq	7(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%rsi, %r14
	callq	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	jmp	.LBB2_4
	.p2align	4, 0x90
.LBB2_17:                               #   in Loop: Header=BB2_4 Depth=1
	xorl	%edx, %edx
.LBB2_18:                               #   in Loop: Header=BB2_4 Depth=1
	movq	%rax, (%rbx,%rdx,8)
	cmpq	$8, %rcx
	jle	.LBB2_38
.LBB2_4:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_7 Depth 2
                                        #     Child Loop BB2_15 Depth 2
	movq	-8(%r14), %rax
	movq	(%rbx), %rdx
	movq	%rdx, -8(%r14)
	addq	$-8, %r14
	movq	%r14, %rcx
	subq	%rbx, %rcx
	movq	%rcx, %rsi
	sarq	$3, %rsi
	cmpq	$3, %rsi
	jl	.LBB2_5
# %bb.6:                                #   in Loop: Header=BB2_4 Depth=1
	leaq	-1(%rsi), %rdx
	shrq	$63, %rdx
	leaq	(%rsi,%rdx), %rdi
	decq	%rdi
	sarq	%rdi
	xorl	%r8d, %r8d
	jmp	.LBB2_7
	.p2align	4, 0x90
.LBB2_9:                                #   in Loop: Header=BB2_7 Depth=2
	leaq	2(,%r8,2), %rdx
.LBB2_10:                               #   in Loop: Header=BB2_7 Depth=2
	movq	(%rbx,%rdx,8), %r9
	movq	%r9, (%rbx,%r8,8)
	movq	%rdx, %r8
	cmpq	%rdi, %rdx
	jge	.LBB2_11
.LBB2_7:                                #   Parent Loop BB2_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r8,%r8), %rdx
	movq	16(%rbx,%rdx,8), %r9
	cmpq	8(%rbx,%rdx,8), %r9
	jae	.LBB2_9
# %bb.8:                                #   in Loop: Header=BB2_7 Depth=2
	leaq	1(,%r8,2), %rdx
	jmp	.LBB2_10
	.p2align	4, 0x90
.LBB2_5:                                #   in Loop: Header=BB2_4 Depth=1
	xorl	%edx, %edx
.LBB2_11:                               #   in Loop: Header=BB2_4 Depth=1
	testb	$8, %cl
	jne	.LBB2_14
# %bb.12:                               #   in Loop: Header=BB2_4 Depth=1
	addq	$-2, %rsi
	sarq	%rsi
	cmpq	%rsi, %rdx
	jne	.LBB2_14
# %bb.13:                               #   in Loop: Header=BB2_4 Depth=1
	leaq	(%rdx,%rdx), %rsi
	movq	8(%rbx,%rsi,8), %rsi
	movq	%rsi, (%rbx,%rdx,8)
	leaq	1(,%rdx,2), %rdx
.LBB2_14:                               #   in Loop: Header=BB2_4 Depth=1
	testq	%rdx, %rdx
	jle	.LBB2_18
	.p2align	4, 0x90
.LBB2_15:                               #   Parent Loop BB2_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rdx), %rsi
	movq	%rsi, %rdi
	shrq	%rdi
	movq	(%rbx,%rdi,8), %r8
	cmpq	%rax, %r8
	jae	.LBB2_18
# %bb.16:                               #   in Loop: Header=BB2_15 Depth=2
	movq	%r8, (%rbx,%rdx,8)
	movq	%rdi, %rdx
	cmpq	$1, %rsi
	ja	.LBB2_15
	jmp	.LBB2_17
.LBB2_38:
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end2-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ # -- Begin function _ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.p2align	4, 0x90
	.type	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,@function
_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_: # @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movq	%rsi, %rax
	subq	%rdi, %rax
	cmpq	$129, %rax
	jl	.LBB3_79
# %bb.1:
	leaq	8(%r14), %rax
	movq	8(%r14), %rcx
	movq	%rax, %rdx
	cmpq	(%r14), %rcx
	jae	.LBB3_3
# %bb.2:
	movq	(%r14), %rdx
	movq	%rdx, 8(%r14)
	movq	%r14, %rdx
.LBB3_3:
	movq	%rcx, (%rdx)
	leaq	16(%r14), %rdx
	movq	16(%r14), %rcx
	cmpq	(%r14), %rcx
	jae	.LBB3_4
# %bb.7:
	vmovups	(%r14), %xmm0
	vmovups	%xmm0, (%rax)
	movq	%r14, %rsi
	jmp	.LBB3_8
.LBB3_79:
	cmpq	%rbx, %r14
	je	.LBB3_88
# %bb.80:
	leaq	8(%r14), %rax
	cmpq	%rbx, %rax
	je	.LBB3_88
# %bb.81:
	movq	%r14, %r15
	jmp	.LBB3_82
	.p2align	4, 0x90
.LBB3_83:                               #   in Loop: Header=BB3_82 Depth=1
	movq	%r15, %rdx
	subq	%r14, %rdx
	subq	%rdx, %rdi
	addq	$16, %rdi
	movq	%r14, %rsi
	callq	memmove@PLT
	movq	%r14, %rax
.LBB3_87:                               #   in Loop: Header=BB3_82 Depth=1
	movq	%r12, (%rax)
	leaq	8(%r15), %rax
	cmpq	%rbx, %rax
	je	.LBB3_88
.LBB3_82:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_86 Depth 2
	movq	%r15, %rdi
	movq	%rax, %r15
	movq	8(%rdi), %r12
	cmpq	(%r14), %r12
	jb	.LBB3_83
# %bb.84:                               #   in Loop: Header=BB3_82 Depth=1
	movq	(%rdi), %rcx
	movq	%r15, %rax
	cmpq	%rcx, %r12
	jae	.LBB3_87
# %bb.85:                               #   in Loop: Header=BB3_82 Depth=1
	movq	%r15, %rax
	.p2align	4, 0x90
.LBB3_86:                               #   Parent Loop BB3_82 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rcx, (%rax)
	movq	-16(%rax), %rcx
	addq	$-8, %rax
	cmpq	%rcx, %r12
	jb	.LBB3_86
	jmp	.LBB3_87
.LBB3_4:
	movq	(%rax), %rdi
	movq	%rdx, %rsi
	cmpq	%rdi, %rcx
	jae	.LBB3_8
# %bb.5:
	movq	%rdx, %rsi
	.p2align	4, 0x90
.LBB3_6:                                # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rsi)
	movq	-16(%rsi), %rdi
	addq	$-8, %rsi
	cmpq	%rdi, %rcx
	jb	.LBB3_6
.LBB3_8:
	movq	%rcx, (%rsi)
	leaq	24(%r14), %rcx
	movq	24(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_9
# %bb.12:
	vmovups	(%r14), %xmm0
	movq	16(%r14), %rdx
	movq	%rdx, 16(%rax)
	vmovups	%xmm0, (%rax)
	movq	%r14, %rdx
	jmp	.LBB3_13
.LBB3_9:
	movq	(%rdx), %rdi
	movq	%rcx, %rdx
	cmpq	%rdi, %rsi
	jae	.LBB3_13
# %bb.10:
	movq	%rcx, %rdx
	.p2align	4, 0x90
.LBB3_11:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rdx)
	movq	-16(%rdx), %rdi
	addq	$-8, %rdx
	cmpq	%rdi, %rsi
	jb	.LBB3_11
.LBB3_13:
	movq	%rsi, (%rdx)
	leaq	32(%r14), %rdx
	movq	32(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_14
# %bb.17:
	vmovups	(%r14), %ymm0
	vmovups	%ymm0, (%rax)
	movq	%r14, %rcx
	jmp	.LBB3_18
.LBB3_14:
	movq	(%rcx), %rdi
	movq	%rdx, %rcx
	cmpq	%rdi, %rsi
	jae	.LBB3_18
# %bb.15:
	movq	%rdx, %rcx
	.p2align	4, 0x90
.LBB3_16:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rcx)
	movq	-16(%rcx), %rdi
	addq	$-8, %rcx
	cmpq	%rdi, %rsi
	jb	.LBB3_16
.LBB3_18:
	movq	%rsi, (%rcx)
	leaq	40(%r14), %rcx
	movq	40(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_19
# %bb.22:
	vmovups	(%r14), %ymm0
	movq	32(%r14), %rdx
	movq	%rdx, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rdx
	jmp	.LBB3_23
.LBB3_19:
	movq	(%rdx), %rdi
	movq	%rcx, %rdx
	cmpq	%rdi, %rsi
	jae	.LBB3_23
# %bb.20:
	movq	%rcx, %rdx
	.p2align	4, 0x90
.LBB3_21:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rdx)
	movq	-16(%rdx), %rdi
	addq	$-8, %rdx
	cmpq	%rdi, %rsi
	jb	.LBB3_21
.LBB3_23:
	movq	%rsi, (%rdx)
	leaq	48(%r14), %rdx
	movq	48(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_24
# %bb.27:
	vmovups	(%r14), %ymm0
	vmovups	32(%r14), %xmm1
	vmovups	%xmm1, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rcx
	jmp	.LBB3_28
.LBB3_24:
	movq	(%rcx), %rdi
	movq	%rdx, %rcx
	cmpq	%rdi, %rsi
	jae	.LBB3_28
# %bb.25:
	movq	%rdx, %rcx
	.p2align	4, 0x90
.LBB3_26:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rcx)
	movq	-16(%rcx), %rdi
	addq	$-8, %rcx
	cmpq	%rdi, %rsi
	jb	.LBB3_26
.LBB3_28:
	movq	%rsi, (%rcx)
	leaq	56(%r14), %rcx
	movq	56(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_29
# %bb.32:
	vmovups	(%r14), %ymm0
	vmovups	32(%r14), %xmm1
	movq	48(%r14), %rdx
	movq	%rdx, 48(%rax)
	vmovups	%xmm1, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rdx
	jmp	.LBB3_33
.LBB3_29:
	movq	(%rdx), %rdi
	movq	%rcx, %rdx
	cmpq	%rdi, %rsi
	jae	.LBB3_33
# %bb.30:
	movq	%rcx, %rdx
	.p2align	4, 0x90
.LBB3_31:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rdx)
	movq	-16(%rdx), %rdi
	addq	$-8, %rdx
	cmpq	%rdi, %rsi
	jb	.LBB3_31
.LBB3_33:
	movq	%rsi, (%rdx)
	leaq	64(%r14), %rdx
	movq	64(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_34
# %bb.37:
	vmovups	(%r14), %ymm0
	vmovups	32(%r14), %ymm1
	vmovups	%ymm1, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rcx
	jmp	.LBB3_38
.LBB3_34:
	movq	(%rcx), %rdi
	movq	%rdx, %rcx
	cmpq	%rdi, %rsi
	jae	.LBB3_38
# %bb.35:
	movq	%rdx, %rcx
	.p2align	4, 0x90
.LBB3_36:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rcx)
	movq	-16(%rcx), %rdi
	addq	$-8, %rcx
	cmpq	%rdi, %rsi
	jb	.LBB3_36
.LBB3_38:
	movq	%rsi, (%rcx)
	leaq	72(%r14), %rcx
	movq	72(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_39
# %bb.42:
	vmovups	(%r14), %ymm0
	vmovups	32(%r14), %ymm1
	movq	64(%r14), %rdx
	movq	%rdx, 64(%rax)
	vmovups	%ymm1, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rdx
	jmp	.LBB3_43
.LBB3_39:
	movq	(%rdx), %rdi
	movq	%rcx, %rdx
	cmpq	%rdi, %rsi
	jae	.LBB3_43
# %bb.40:
	movq	%rcx, %rdx
	.p2align	4, 0x90
.LBB3_41:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rdx)
	movq	-16(%rdx), %rdi
	addq	$-8, %rdx
	cmpq	%rdi, %rsi
	jb	.LBB3_41
.LBB3_43:
	movq	%rsi, (%rdx)
	leaq	80(%r14), %rdx
	movq	80(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_44
# %bb.47:
	vmovups	64(%r14), %xmm0
	vmovups	(%r14), %ymm1
	vmovups	32(%r14), %ymm2
	vmovups	%ymm2, 32(%rax)
	vmovups	%xmm0, 64(%rax)
	vmovups	%ymm1, (%rax)
	movq	%r14, %rcx
	jmp	.LBB3_48
.LBB3_44:
	movq	(%rcx), %rdi
	movq	%rdx, %rcx
	cmpq	%rdi, %rsi
	jae	.LBB3_48
# %bb.45:
	movq	%rdx, %rcx
	.p2align	4, 0x90
.LBB3_46:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rcx)
	movq	-16(%rcx), %rdi
	addq	$-8, %rcx
	cmpq	%rdi, %rsi
	jb	.LBB3_46
.LBB3_48:
	movq	%rsi, (%rcx)
	leaq	88(%r14), %rcx
	movq	88(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_49
# %bb.52:
	vmovups	64(%r14), %xmm0
	movq	80(%r14), %rdx
	vmovups	(%r14), %ymm1
	vmovups	32(%r14), %ymm2
	vmovups	%ymm2, 32(%rax)
	movq	%rdx, 80(%rax)
	vmovups	%xmm0, 64(%rax)
	vmovups	%ymm1, (%rax)
	movq	%r14, %rdx
	jmp	.LBB3_53
.LBB3_49:
	movq	(%rdx), %rdi
	movq	%rcx, %rdx
	cmpq	%rdi, %rsi
	jae	.LBB3_53
# %bb.50:
	movq	%rcx, %rdx
	.p2align	4, 0x90
.LBB3_51:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rdx)
	movq	-16(%rdx), %rdi
	addq	$-8, %rdx
	cmpq	%rdi, %rsi
	jb	.LBB3_51
.LBB3_53:
	movq	%rsi, (%rdx)
	leaq	96(%r14), %rdx
	movq	96(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_54
# %bb.57:
	vmovups	(%r14), %ymm0
	vmovups	32(%r14), %ymm1
	vmovups	64(%r14), %ymm2
	vmovups	%ymm2, 64(%rax)
	vmovups	%ymm1, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rcx
	jmp	.LBB3_58
.LBB3_54:
	movq	(%rcx), %rdi
	movq	%rdx, %rcx
	cmpq	%rdi, %rsi
	jae	.LBB3_58
# %bb.55:
	movq	%rdx, %rcx
	.p2align	4, 0x90
.LBB3_56:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rcx)
	movq	-16(%rcx), %rdi
	addq	$-8, %rcx
	cmpq	%rdi, %rsi
	jb	.LBB3_56
.LBB3_58:
	movq	%rsi, (%rcx)
	leaq	104(%r14), %rcx
	movq	104(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_59
# %bb.62:
	vmovups	(%r14), %ymm0
	vmovups	32(%r14), %ymm1
	vmovups	64(%r14), %ymm2
	movq	96(%r14), %rdx
	movq	%rdx, 96(%rax)
	vmovups	%ymm2, 64(%rax)
	vmovups	%ymm1, 32(%rax)
	vmovups	%ymm0, (%rax)
	movq	%r14, %rdx
	jmp	.LBB3_63
.LBB3_59:
	movq	(%rdx), %rdi
	movq	%rcx, %rdx
	cmpq	%rdi, %rsi
	jae	.LBB3_63
# %bb.60:
	movq	%rcx, %rdx
	.p2align	4, 0x90
.LBB3_61:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rdx)
	movq	-16(%rdx), %rdi
	addq	$-8, %rdx
	cmpq	%rdi, %rsi
	jb	.LBB3_61
.LBB3_63:
	movq	%rsi, (%rdx)
	leaq	112(%r14), %rdx
	movq	112(%r14), %rsi
	cmpq	(%r14), %rsi
	jae	.LBB3_64
# %bb.67:
	vmovups	96(%r14), %xmm0
	vmovups	(%r14), %ymm1
	vmovups	32(%r14), %ymm2
	vmovups	64(%r14), %ymm3
	vmovups	%ymm2, 32(%rax)
	vmovups	%ymm3, 64(%rax)
	vmovups	%xmm0, 96(%rax)
	vmovups	%ymm1, (%rax)
	movq	%r14, %rcx
	jmp	.LBB3_68
.LBB3_64:
	movq	(%rcx), %rdi
	movq	%rdx, %rcx
	cmpq	%rdi, %rsi
	jae	.LBB3_68
# %bb.65:
	movq	%rdx, %rcx
	.p2align	4, 0x90
.LBB3_66:                               # =>This Inner Loop Header: Depth=1
	movq	%rdi, (%rcx)
	movq	-16(%rcx), %rdi
	addq	$-8, %rcx
	cmpq	%rdi, %rsi
	jb	.LBB3_66
.LBB3_68:
	movq	%rsi, (%rcx)
	movq	120(%r14), %rcx
	cmpq	(%r14), %rcx
	jae	.LBB3_69
# %bb.72:
	vmovups	96(%r14), %xmm0
	movq	112(%r14), %rdx
	vmovups	(%r14), %ymm1
	vmovups	32(%r14), %ymm2
	vmovups	64(%r14), %ymm3
	vmovups	%ymm2, 32(%rax)
	vmovups	%ymm3, 64(%rax)
	movq	%rdx, 112(%rax)
	vmovups	%xmm0, 96(%rax)
	vmovups	%ymm1, (%rax)
	movq	%r14, %rax
.LBB3_73:
	movq	%rcx, (%rax)
	subq	$-128, %r14
	jmp	.LBB3_74
	.p2align	4, 0x90
.LBB3_78:                               #   in Loop: Header=BB3_74 Depth=1
	movq	%rax, (%rdx)
	addq	$8, %r14
.LBB3_74:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_77 Depth 2
	cmpq	%rbx, %r14
	je	.LBB3_88
# %bb.75:                               #   in Loop: Header=BB3_74 Depth=1
	movq	-8(%r14), %rcx
	movq	(%r14), %rax
	movq	%r14, %rdx
	cmpq	%rcx, %rax
	jae	.LBB3_78
# %bb.76:                               #   in Loop: Header=BB3_74 Depth=1
	movq	%r14, %rdx
	.p2align	4, 0x90
.LBB3_77:                               #   Parent Loop BB3_74 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rcx, (%rdx)
	movq	-16(%rdx), %rcx
	addq	$-8, %rdx
	cmpq	%rcx, %rax
	jb	.LBB3_77
	jmp	.LBB3_78
.LBB3_88:
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB3_69:
	.cfi_def_cfa_offset 48
	leaq	120(%r14), %rax
	movq	(%rdx), %rdx
	cmpq	%rdx, %rcx
	jae	.LBB3_73
	.p2align	4, 0x90
.LBB3_71:                               # =>This Inner Loop Header: Depth=1
	movq	%rdx, (%rax)
	movq	-16(%rax), %rdx
	addq	$-8, %rax
	cmpq	%rdx, %rcx
	jb	.LBB3_71
	jmp	.LBB3_73
.Lfunc_end3:
	.size	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_, .Lfunc_end3-_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,"axG",@progbits,_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,comdat
	.weak	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ # -- Begin function _ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.p2align	4, 0x90
	.type	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_,@function
_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_: # @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_startproc
# %bb.0:
	subq	%rdi, %rsi
	movq	%rsi, %rax
	sarq	$3, %rax
	cmpq	$2, %rax
	jge	.LBB4_2
.LBB4_1:
	retq
.LBB4_2:
	leaq	-2(%rax), %rdx
	movq	%rdx, %rcx
	shrq	%rcx
	decq	%rax
	shrq	%rax
	testb	$8, %sil
	jne	.LBB4_20
# %bb.3:
	orq	$1, %rdx
	movq	%rcx, %rsi
	jmp	.LBB4_6
	.p2align	4, 0x90
.LBB4_4:                                #   in Loop: Header=BB4_6 Depth=1
	movq	%r9, %r10
.LBB4_5:                                #   in Loop: Header=BB4_6 Depth=1
	movq	%r8, (%rdi,%r10,8)
	subq	$1, %rsi
	jb	.LBB4_1
.LBB4_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_10 Depth 2
                                        #     Child Loop BB4_15 Depth 2
	movq	(%rdi,%rsi,8), %r8
	movq	%rsi, %r9
	cmpq	%rsi, %rax
	jle	.LBB4_12
# %bb.7:                                #   in Loop: Header=BB4_6 Depth=1
	movq	%rsi, %r10
	jmp	.LBB4_10
	.p2align	4, 0x90
.LBB4_8:                                #   in Loop: Header=BB4_10 Depth=2
	leaq	2(,%r10,2), %r9
.LBB4_9:                                #   in Loop: Header=BB4_10 Depth=2
	movq	(%rdi,%r9,8), %r11
	movq	%r11, (%rdi,%r10,8)
	movq	%r9, %r10
	cmpq	%rax, %r9
	jge	.LBB4_12
.LBB4_10:                               #   Parent Loop BB4_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r10,%r10), %r9
	movq	16(%rdi,%r9,8), %r11
	cmpq	8(%rdi,%r9,8), %r11
	jae	.LBB4_8
# %bb.11:                               #   in Loop: Header=BB4_10 Depth=2
	leaq	1(,%r10,2), %r9
	jmp	.LBB4_9
	.p2align	4, 0x90
.LBB4_12:                               #   in Loop: Header=BB4_6 Depth=1
	cmpq	%rcx, %r9
	jne	.LBB4_14
# %bb.13:                               #   in Loop: Header=BB4_6 Depth=1
	movq	(%rdi,%rdx,8), %r9
	movq	%r9, (%rdi,%rcx,8)
	movq	%rdx, %r9
.LBB4_14:                               #   in Loop: Header=BB4_6 Depth=1
	cmpq	%rsi, %r9
	jle	.LBB4_4
	.p2align	4, 0x90
.LBB4_15:                               #   Parent Loop BB4_6 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%r9), %r10
	shrq	$63, %r10
	addq	%r9, %r10
	decq	%r10
	sarq	%r10
	movq	(%rdi,%r10,8), %r11
	cmpq	%r8, %r11
	jae	.LBB4_4
# %bb.16:                               #   in Loop: Header=BB4_15 Depth=2
	movq	%r11, (%rdi,%r9,8)
	movq	%r10, %r9
	cmpq	%rsi, %r10
	jg	.LBB4_15
	jmp	.LBB4_5
	.p2align	4, 0x90
.LBB4_18:                               #   in Loop: Header=BB4_20 Depth=1
	movq	%rsi, %r8
.LBB4_19:                               #   in Loop: Header=BB4_20 Depth=1
	movq	%rdx, (%rdi,%r8,8)
	subq	$1, %rcx
	jb	.LBB4_1
.LBB4_20:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_24 Depth 2
                                        #     Child Loop BB4_27 Depth 2
	movq	(%rdi,%rcx,8), %rdx
	movq	%rcx, %r8
	cmpq	%rcx, %rax
	jle	.LBB4_19
# %bb.21:                               #   in Loop: Header=BB4_20 Depth=1
	movq	%rcx, %r8
	jmp	.LBB4_24
	.p2align	4, 0x90
.LBB4_22:                               #   in Loop: Header=BB4_24 Depth=2
	leaq	2(,%r8,2), %rsi
.LBB4_23:                               #   in Loop: Header=BB4_24 Depth=2
	movq	(%rdi,%rsi,8), %r9
	movq	%r9, (%rdi,%r8,8)
	movq	%rsi, %r8
	cmpq	%rax, %rsi
	jge	.LBB4_26
.LBB4_24:                               #   Parent Loop BB4_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r8,%r8), %rsi
	movq	16(%rdi,%rsi,8), %r9
	cmpq	8(%rdi,%rsi,8), %r9
	jae	.LBB4_22
# %bb.25:                               #   in Loop: Header=BB4_24 Depth=2
	leaq	1(,%r8,2), %rsi
	jmp	.LBB4_23
	.p2align	4, 0x90
.LBB4_26:                               #   in Loop: Header=BB4_20 Depth=1
	cmpq	%rcx, %rsi
	jle	.LBB4_18
	.p2align	4, 0x90
.LBB4_27:                               #   Parent Loop BB4_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rsi), %r8
	shrq	$63, %r8
	addq	%rsi, %r8
	decq	%r8
	sarq	%r8
	movq	(%rdi,%r8,8), %r9
	cmpq	%rdx, %r9
	jae	.LBB4_18
# %bb.28:                               #   in Loop: Header=BB4_27 Depth=2
	movq	%r9, (%rdi,%rsi,8)
	movq	%r8, %rsi
	cmpq	%rcx, %r8
	jg	.LBB4_27
	jmp	.LBB4_19
.Lfunc_end4:
	.size	_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_, .Lfunc_end4-_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_memcpy_test.cpp
	.type	_GLOBAL__sub_I_memcpy_test.cpp,@function
_GLOBAL__sub_I_memcpy_test.cpp:         # @_GLOBAL__sub_I_memcpy_test.cpp
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	leaq	_ZStL8__ioinit(%rip), %rbx
	movq	%rbx, %rdi
	callq	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	__dso_handle(%rip), %rdx
	movq	%rbx, %rsi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit@PLT                # TAILCALL
.Lfunc_end5:
	.size	_GLOBAL__sub_I_memcpy_test.cpp, .Lfunc_end5-_GLOBAL__sub_I_memcpy_test.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	" ops/s"
	.size	.L.str, 7

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	" ns/op"
	.size	.L.str.1, 7

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"vector::_M_realloc_insert"
	.size	.L.str.2, 26

	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_memcpy_test.cpp
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"Ubuntu clang version 19.0.0 (++20240222053038+26d71d9ed56c-1~exp1~20240222173217.940)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _GLOBAL__sub_I_memcpy_test.cpp
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
