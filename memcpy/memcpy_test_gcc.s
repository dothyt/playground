	.file	"memcpy_test.cpp"
	.text
	.p2align 4
	.type	_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElmNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0, @function
_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElmNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0:
.LFB8603:
	.cfi_startproc
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	leaq	-1(%rdx), %rax
	movq	%rdx, %r13
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	movq	%rsi, %r8
	movq	%rdx, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movq	%rax, %rbp
	shrq	$63, %rbp
	addq	%rax, %rbp
	sarq	%rbp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rcx, %r9
	andl	$1, %r13d
	cmpq	%rbp, %rsi
	jl	.L6
	jmp	.L2
	.p2align 4,,10
	.p2align 3
.L18:
	movq	%r11, (%rdi,%rsi,8)
	cmpq	%rdx, %rbp
	jle	.L4
.L5:
	movq	%rdx, %rsi
.L6:
	leaq	1(%rsi), %rax
	leaq	(%rax,%rax), %rdx
	leaq	-1(%rdx), %rcx
	salq	$4, %rax
	leaq	(%rdi,%rcx,8), %rbx
	addq	%rdi, %rax
	movq	(%rax), %r11
	movq	(%rbx), %r10
	cmpq	%r11, %r10
	jbe	.L18
	movq	%r10, (%rdi,%rsi,8)
	cmpq	%rcx, %rbp
	jle	.L11
	movq	%rcx, %rdx
	jmp	.L5
	.p2align 4,,10
	.p2align 3
.L11:
	movq	%rbx, %rax
	movq	%rcx, %rdx
.L4:
	testq	%r13, %r13
	je	.L10
.L7:
	leaq	-1(%rdx), %rsi
	movq	%rsi, %rcx
	shrq	$63, %rcx
	addq	%rsi, %rcx
	sarq	%rcx
	cmpq	%r8, %rdx
	jg	.L9
	jmp	.L8
	.p2align 4,,10
	.p2align 3
.L20:
	leaq	-1(%rcx), %rdx
	movq	%rsi, (%rax)
	movq	%rdx, %rax
	shrq	$63, %rax
	addq	%rdx, %rax
	sarq	%rax
	movq	%rcx, %rdx
	cmpq	%rcx, %r8
	jge	.L19
	movq	%rax, %rcx
.L9:
	leaq	(%rdi,%rcx,8), %r10
	movq	(%r10), %rsi
	leaq	(%rdi,%rdx,8), %rax
	cmpq	%r9, %rsi
	jb	.L20
.L8:
	movq	%r9, (%rax)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2:
	.cfi_restore_state
	leaq	(%rdi,%rsi,8), %rax
	testq	%r13, %r13
	jne	.L8
	movq	%r8, %rdx
	.p2align 4,,10
	.p2align 3
.L10:
	leaq	-2(%r12), %rcx
	movq	%rcx, %r12
	shrq	$63, %r12
	addq	%rcx, %r12
	sarq	%r12
	cmpq	%r12, %rdx
	jne	.L7
	leaq	1(%rdx,%rdx), %rdx
	leaq	(%rdi,%rdx,8), %rcx
	movq	(%rcx), %rsi
	movq	%rsi, (%rax)
	movq	%rcx, %rax
	jmp	.L7
	.p2align 4,,10
	.p2align 3
.L19:
	movq	%r10, %rax
	movq	%r9, (%rax)
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE8603:
	.size	_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElmNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0, .-_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElmNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0
	.p2align 4
	.type	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0, @function
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0:
.LFB8596:
	.cfi_startproc
	movq	%rsi, %rax
	subq	%rdi, %rax
	cmpq	$128, %rax
	jle	.L51
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	movq	%rdx, %r14
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	movq	%rdi, %r13
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	testq	%rdx, %rdx
	je	.L23
	movq	%rsi, %r8
	leaq	8(%rdi), %rbp
.L25:
	movq	%r8, %rax
	subq	%r13, %rax
	movq	%rax, %rdx
	sarq	$3, %rdx
	shrq	$63, %rax
	addq	%rdx, %rax
	sarq	%rax
	leaq	0(%r13,%rax,8), %rdi
	movq	8(%r13), %rcx
	movq	(%rdi), %rax
	movq	-8(%r8), %rsi
	movq	0(%r13), %rdx
	decq	%r14
	cmpq	%rax, %rcx
	jnb	.L30
	cmpq	%rsi, %rax
	jb	.L35
	cmpq	%rsi, %rcx
	jnb	.L33
.L54:
	movq	%rsi, 0(%r13)
	movq	%rdx, -8(%r8)
	movq	0(%r13), %rcx
.L32:
	movq	%rbp, %rbx
	movq	%r8, %rax
	.p2align 4,,10
	.p2align 3
.L36:
	movq	(%rbx), %rsi
	movq	%rbx, %r12
	cmpq	%rcx, %rsi
	jb	.L37
	subq	$8, %rax
	cmpq	%rcx, %rdx
	jbe	.L38
	.p2align 4,,10
	.p2align 3
.L39:
	movq	-8(%rax), %rdx
	subq	$8, %rax
	cmpq	%rcx, %rdx
	ja	.L39
.L38:
	cmpq	%rbx, %rax
	jbe	.L55
	movq	%rdx, (%rbx)
	movq	%rsi, (%rax)
	movq	-8(%rax), %rdx
	movq	0(%r13), %rcx
.L37:
	addq	$8, %rbx
	jmp	.L36
	.p2align 4,,10
	.p2align 3
.L55:
	movq	%r14, %rdx
	movq	%r8, %rsi
	movq	%rbx, %rdi
	call	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0
	movq	%rbx, %rax
	subq	%r13, %rax
	cmpq	$128, %rax
	jle	.L49
	testq	%r14, %r14
	je	.L23
	movq	%rbx, %r8
	jmp	.L25
.L30:
	cmpq	%rsi, %rcx
	jnb	.L34
	vmovq	%rcx, %xmm1
	vpinsrq	$1, %rdx, %xmm1, %xmm0
	vmovdqu	%xmm0, 0(%r13)
	movq	-8(%r8), %rdx
	jmp	.L32
.L34:
	cmpq	%rsi, %rax
	jb	.L54
.L35:
	movq	%rax, 0(%r13)
	movq	%rdx, (%rdi)
	movq	0(%r13), %rcx
	movq	-8(%r8), %rdx
	jmp	.L32
.L23:
	sarq	$3, %rax
	leaq	-2(%rax), %rbp
	movq	%rax, %rbx
	sarq	%rbp
	jmp	.L27
.L56:
	decq	%rbp
.L27:
	movq	0(%r13,%rbp,8), %rcx
	movq	%rbx, %rdx
	movq	%rbp, %rsi
	movq	%r13, %rdi
	call	_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElmNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0
	testq	%rbp, %rbp
	jne	.L56
	subq	$8, %r12
	.p2align 4,,10
	.p2align 3
.L28:
	movq	0(%r13), %rax
	movq	%r12, %rbx
	movq	(%r12), %rcx
	subq	%r13, %rbx
	movq	%rax, (%r12)
	movq	%rbx, %rdx
	sarq	$3, %rdx
	xorl	%esi, %esi
	movq	%r13, %rdi
	call	_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElmNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0
	subq	$8, %r12
	cmpq	$8, %rbx
	jg	.L28
.L49:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
.L33:
	.cfi_restore_state
	vmovq	%rcx, %xmm2
	vpinsrq	$1, %rdx, %xmm2, %xmm0
	vmovdqu	%xmm0, 0(%r13)
	movq	-8(%r8), %rdx
	jmp	.L32
.L51:
	.cfi_def_cfa_offset 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	ret
	.cfi_endproc
.LFE8596:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0, .-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0
	.p2align 4
	.type	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, @function
_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0:
.LFB8591:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movq	(%rdi), %rax
	movq	-24(%rax), %rax
	movq	240(%rdi,%rax), %r12
	testq	%r12, %r12
	je	.L62
	cmpb	$0, 56(%r12)
	movq	%rdi, %rbp
	je	.L59
	movsbl	67(%r12), %esi
.L60:
	movq	%rbp, %rdi
	call	_ZNSo3putEc@PLT
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rax, %rdi
	popq	%r12
	.cfi_def_cfa_offset 8
	jmp	_ZNSo5flushEv@PLT
.L59:
	.cfi_restore_state
	movq	%r12, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	(%r12), %rax
	movl	$10, %esi
	movq	%r12, %rdi
	call	*48(%rax)
	movsbl	%al, %esi
	jmp	.L60
.L62:
	call	_ZSt16__throw_bad_castv@PLT
	.cfi_endproc
.LFE8591:
	.size	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, .-_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	.p2align 4
	.globl	_Z13avx512_memcpyPvS_m
	.type	_Z13avx512_memcpyPvS_m, @function
_Z13avx512_memcpyPvS_m:
.LFB7925:
	.cfi_startproc
	endbr64
	cmpq	$512, %rdx
	jbe	.L67
	movl	$512, %ecx
	subq	%rdi, %rcx
	.p2align 4,,10
	.p2align 3
.L65:
	vmovdqa64	64(%rsi), %zmm6
	vmovdqa64	128(%rsi), %zmm5
	vmovdqa64	192(%rsi), %zmm4
	vmovdqa64	256(%rsi), %zmm3
	vmovdqa64	320(%rsi), %zmm2
	vmovdqa64	384(%rsi), %zmm1
	vmovdqa64	448(%rsi), %zmm0
	vmovdqa64	(%rsi), %zmm7
	prefetcht1	(%rdi)
	prefetcht1	64(%rdi)
	vmovdqa64	%zmm7, (%rdi)
	vmovdqa64	%zmm6, 64(%rdi)
	prefetcht1	128(%rdi)
	prefetcht1	192(%rdi)
	vmovdqa64	%zmm5, 128(%rdi)
	vmovdqa64	%zmm4, 192(%rdi)
	prefetcht1	256(%rdi)
	prefetcht1	320(%rdi)
	vmovdqa64	%zmm3, 256(%rdi)
	vmovdqa64	%zmm2, 320(%rdi)
	prefetcht1	384(%rdi)
	prefetcht1	448(%rdi)
	vmovdqa64	%zmm1, 384(%rdi)
	vmovdqa64	%zmm0, 448(%rdi)
	addq	$512, %rdi
	leaq	(%rcx,%rdi), %rax
	addq	$512, %rsi
	cmpq	%rax, %rdx
	ja	.L65
	vzeroupper
.L67:
	ret
	.cfi_endproc
.LFE7925:
	.size	_Z13avx512_memcpyPvS_m, .-_Z13avx512_memcpyPvS_m
	.p2align 4
	.globl	_Z18avx512_memcpy_1024PvS_m
	.type	_Z18avx512_memcpy_1024PvS_m, @function
_Z18avx512_memcpy_1024PvS_m:
.LFB7926:
	.cfi_startproc
	endbr64
	cmpq	$1024, %rdx
	jbe	.L72
	movl	$1024, %ecx
	subq	%rdi, %rcx
	.p2align 4,,10
	.p2align 3
.L70:
	vmovdqa64	64(%rsi), %zmm14
	vmovdqa64	128(%rsi), %zmm13
	vmovdqa64	192(%rsi), %zmm12
	vmovdqa64	256(%rsi), %zmm11
	vmovdqa64	320(%rsi), %zmm10
	vmovdqa64	384(%rsi), %zmm9
	vmovdqa64	448(%rsi), %zmm8
	vmovdqa64	512(%rsi), %zmm7
	vmovdqa64	576(%rsi), %zmm6
	vmovdqa64	640(%rsi), %zmm5
	vmovdqa64	704(%rsi), %zmm4
	vmovdqa64	768(%rsi), %zmm3
	vmovdqa64	832(%rsi), %zmm2
	vmovdqa64	896(%rsi), %zmm1
	vmovdqa64	960(%rsi), %zmm0
	vmovdqa64	(%rsi), %zmm15
	prefetcht1	(%rdi)
	prefetcht1	64(%rdi)
	prefetcht1	128(%rdi)
	prefetcht1	192(%rdi)
	prefetcht1	256(%rdi)
	prefetcht1	320(%rdi)
	prefetcht1	384(%rdi)
	prefetcht1	448(%rdi)
	prefetcht1	512(%rdi)
	prefetcht1	576(%rdi)
	prefetcht1	640(%rdi)
	prefetcht1	704(%rdi)
	prefetcht1	768(%rdi)
	prefetcht1	832(%rdi)
	prefetcht1	896(%rdi)
	prefetcht1	960(%rdi)
	vmovdqa64	%zmm15, (%rdi)
	vmovdqa64	%zmm14, 64(%rdi)
	vmovdqa64	%zmm13, 128(%rdi)
	vmovdqa64	%zmm12, 192(%rdi)
	vmovdqa64	%zmm11, 256(%rdi)
	vmovdqa64	%zmm10, 320(%rdi)
	vmovdqa64	%zmm9, 384(%rdi)
	vmovdqa64	%zmm8, 448(%rdi)
	vmovdqa64	%zmm7, 512(%rdi)
	vmovdqa64	%zmm6, 576(%rdi)
	vmovdqa64	%zmm5, 640(%rdi)
	vmovdqa64	%zmm4, 704(%rdi)
	vmovdqa64	%zmm3, 768(%rdi)
	vmovdqa64	%zmm2, 832(%rdi)
	vmovdqa64	%zmm1, 896(%rdi)
	vmovdqa64	%zmm0, 960(%rdi)
	addq	$1024, %rdi
	leaq	(%rcx,%rdi), %rax
	addq	$1024, %rsi
	cmpq	%rax, %rdx
	ja	.L70
	vzeroupper
.L72:
	ret
	.cfi_endproc
.LFE7926:
	.size	_Z18avx512_memcpy_1024PvS_m, .-_Z18avx512_memcpy_1024PvS_m
	.p2align 4
	.globl	_Z13avx512_memcpyPvS_S_m
	.type	_Z13avx512_memcpyPvS_S_m, @function
_Z13avx512_memcpyPvS_S_m:
.LFB7927:
	.cfi_startproc
	endbr64
	cmpq	$512, %rcx
	jbe	.L77
	movl	$512, %r8d
	subq	%rdx, %r8
	.p2align 4,,10
	.p2align 3
.L75:
	vmovdqa64	(%rdx), %zmm7
	vmovdqa64	64(%rdx), %zmm6
	vmovdqa64	128(%rdx), %zmm5
	vmovdqa64	192(%rdx), %zmm4
	vmovdqa64	256(%rdx), %zmm3
	vmovdqa64	320(%rdx), %zmm2
	vmovdqa64	384(%rdx), %zmm1
	vmovdqa64	448(%rdx), %zmm0
	vmovdqa64	64(%rsi), %zmm14
	vmovdqa64	128(%rsi), %zmm13
	vmovdqa64	192(%rsi), %zmm12
	vmovdqa64	256(%rsi), %zmm11
	vmovdqa64	320(%rsi), %zmm10
	vmovdqa64	384(%rsi), %zmm9
	vmovdqa64	448(%rsi), %zmm8
	vmovdqa64	(%rsi), %zmm15
	prefetcht1	(%rdx)
	prefetcht1	64(%rdx)
	prefetcht1	128(%rdx)
	prefetcht1	192(%rdx)
	prefetcht1	256(%rdx)
	prefetcht1	320(%rdx)
	prefetcht1	384(%rdx)
	prefetcht1	448(%rdx)
	addq	$512, %rdx
	leaq	(%r8,%rdx), %rax
	prefetcht1	(%rdi)
	prefetcht1	64(%rdi)
	prefetcht1	128(%rdi)
	prefetcht1	192(%rdi)
	prefetcht1	256(%rdi)
	prefetcht1	320(%rdi)
	prefetcht1	384(%rdi)
	prefetcht1	448(%rdi)
	addq	$512, %rsi
	vmovdqa64	%zmm15, (%rdi)
	vmovdqa64	%zmm14, 64(%rdi)
	vmovdqa64	%zmm13, 128(%rdi)
	vmovdqa64	%zmm12, 192(%rdi)
	vmovdqa64	%zmm11, 256(%rdi)
	vmovdqa64	%zmm10, 320(%rdi)
	vmovdqa64	%zmm9, 384(%rdi)
	vmovdqa64	%zmm8, 448(%rdi)
	vmovdqa64	%zmm7, -512(%rsi)
	vmovdqa64	%zmm6, -448(%rsi)
	vmovdqa64	%zmm5, -384(%rsi)
	vmovdqa64	%zmm4, -320(%rsi)
	vmovdqa64	%zmm3, -256(%rsi)
	vmovdqa64	%zmm2, -192(%rsi)
	vmovdqa64	%zmm1, -128(%rsi)
	vmovdqa64	%zmm0, -64(%rsi)
	addq	$512, %rdi
	cmpq	%rax, %rcx
	ja	.L75
	vzeroupper
.L77:
	ret
	.cfi_endproc
.LFE7927:
	.size	_Z13avx512_memcpyPvS_S_m, .-_Z13avx512_memcpyPvS_S_m
	.section	.rodata._ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_.str1.1,"aMS",@progbits,1
.LC0:
	.string	"vector::_M_realloc_insert"
	.section	.text._ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_,"axG",@progbits,_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
	.type	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_, @function
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_:
.LFB8349:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movabsq	$1152921504606846975, %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	movq	8(%rdi), %r13
	movq	(%rdi), %r15
	movq	%r13, %rax
	subq	%r15, %rax
	sarq	$3, %rax
	cmpq	%rbx, %rax
	je	.L99
	movq	%rdx, %rcx
	movl	$1, %edx
	cmpq	%rdx, %rax
	cmovnb	%rax, %rdx
	movq	%rsi, %r14
	xorl	%esi, %esi
	addq	%rdx, %rax
	setc	%sil
	movq	%r14, %rdx
	movq	%rdi, %r12
	subq	%r15, %rdx
	testq	%rsi, %rsi
	jne	.L91
	testq	%rax, %rax
	jne	.L83
	xorl	%ebx, %ebx
	xorl	%ebp, %ebp
.L89:
	movq	%r13, %r8
	movq	(%rcx), %rax
	leaq	8(%rbp,%rdx), %r9
	subq	%r14, %r8
	movq	%rax, 0(%rbp,%rdx)
	leaq	(%r9,%r8), %r13
	testq	%rdx, %rdx
	jg	.L100
	testq	%r8, %r8
	jg	.L87
.L86:
	testq	%r15, %r15
	jne	.L98
.L88:
	movq	%rbx, 16(%r12)
	vmovq	%rbp, %xmm1
	vpinsrq	$1, %r13, %xmm1, %xmm0
	vmovdqu	%xmm0, (%r12)
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L83:
	.cfi_restore_state
	cmpq	%rbx, %rax
	cmova	%rbx, %rax
	leaq	0(,%rax,8), %rbx
.L82:
	movq	%rbx, %rdi
	movq	%rcx, 8(%rsp)
	movq	%rdx, (%rsp)
	call	_Znwm@PLT
	movq	%rax, %rbp
	movq	(%rsp), %rdx
	movq	8(%rsp), %rcx
	addq	%rax, %rbx
	jmp	.L89
	.p2align 4,,10
	.p2align 3
.L100:
	movq	%r15, %rsi
	movq	%rbp, %rdi
	movq	%r8, 8(%rsp)
	movq	%r9, (%rsp)
	call	memmove@PLT
	movq	8(%rsp), %r8
	movq	(%rsp), %r9
	testq	%r8, %r8
	jg	.L87
.L98:
	movq	16(%r12), %rsi
	movq	%r15, %rdi
	subq	%r15, %rsi
	call	_ZdlPvm@PLT
	jmp	.L88
	.p2align 4,,10
	.p2align 3
.L87:
	movq	%r8, %rdx
	movq	%r14, %rsi
	movq	%r9, %rdi
	call	memcpy@PLT
	jmp	.L86
	.p2align 4,,10
	.p2align 3
.L91:
	movabsq	$9223372036854775800, %rbx
	jmp	.L82
.L99:
	leaq	.LC0(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE8349:
	.size	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_, .-_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	" ops/s"
.LC2:
	.string	" ns/op"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB3:
	.section	.text.startup,"ax",@progbits
.LHOTB3:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB7928:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA7928
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$4096000000, %esi
	movl	$64, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	movl	$20, %r13d
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$64, %rsp
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	call	aligned_alloc@PLT
	movl	$4096000000, %esi
	movl	$64, %edi
	movq	%rax, %rbx
	call	aligned_alloc@PLT
	movl	$4096000000, %esi
	movl	$64, %edi
	movq	%rax, %r12
	call	aligned_alloc@PLT
	movq	$0, 16(%rsp)
	movq	$0, 24(%rsp)
	movq	$0, 32(%rsp)
	movq	$0, 40(%rsp)
	leaq	3584(%rbx), %r14
	movl	$4096000000, %ebx
	movq	%rax, %r15
	addq	%r12, %rbx
.L106:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, 16(%rsp)
	movq	%r15, %r8
	movq	%r12, %rdi
	movq	%r14, %rsi
	.p2align 4,,10
	.p2align 3
.L103:
	leaq	-3584(%rsi), %rax
	movq	%rdi, %rcx
	movq	%r8, %rdx
	.p2align 4,,10
	.p2align 3
.L102:
	vmovdqa64	64(%rdx), %zmm1
	vmovdqa64	128(%rdx), %zmm2
	vmovdqa64	192(%rdx), %zmm3
	vmovdqa64	256(%rdx), %zmm4
	vmovdqa64	320(%rdx), %zmm5
	vmovdqa64	384(%rdx), %zmm6
	vmovdqa64	448(%rdx), %zmm7
	vmovdqa64	(%rdx), %zmm0
	prefetcht1	(%rax)
	prefetcht1	64(%rax)
	prefetcht1	128(%rax)
	prefetcht1	192(%rax)
	prefetcht1	256(%rax)
	prefetcht1	320(%rax)
	prefetcht1	384(%rax)
	prefetcht1	448(%rax)
	prefetcht1	(%rcx)
	prefetcht1	64(%rcx)
	prefetcht1	128(%rcx)
	vmovdqa64	%zmm1, 64(%rcx)
	vmovdqa64	%zmm2, 128(%rcx)
	prefetcht1	192(%rcx)
	prefetcht1	256(%rcx)
	vmovdqa64	%zmm3, 192(%rcx)
	vmovdqa64	%zmm4, 256(%rcx)
	prefetcht1	320(%rcx)
	prefetcht1	384(%rcx)
	vmovdqa64	%zmm5, 320(%rcx)
	vmovdqa64	%zmm6, 384(%rcx)
	prefetcht1	448(%rcx)
	vmovdqa64	%zmm0, (%rcx)
	vmovdqa64	%zmm7, 448(%rcx)
	vmovdqa64	(%rax), %zmm0
	vmovdqa64	64(%rax), %zmm1
	vmovdqa64	128(%rax), %zmm2
	vmovdqa64	192(%rax), %zmm3
	vmovdqa64	256(%rax), %zmm4
	vmovdqa64	320(%rax), %zmm5
	vmovdqa64	384(%rax), %zmm6
	vmovdqa64	448(%rax), %zmm7
	addq	$512, %rax
	vmovdqa64	%zmm0, (%rdx)
	vmovdqa64	%zmm1, 64(%rdx)
	vmovdqa64	%zmm2, 128(%rdx)
	vmovdqa64	%zmm3, 192(%rdx)
	vmovdqa64	%zmm4, 256(%rdx)
	vmovdqa64	%zmm5, 320(%rdx)
	vmovdqa64	%zmm6, 384(%rdx)
	vmovdqa64	%zmm7, 448(%rdx)
	addq	$512, %rcx
	addq	$512, %rdx
	cmpq	%rax, %rsi
	jne	.L102
	addq	$4096, %rdi
	addq	$4096, %r8
	addq	$4096, %rsi
	cmpq	%rdi, %rbx
	jne	.L103
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	32(%rsp), %rsi
	subq	16(%rsp), %rax
	movq	%rax, 8(%rsp)
	cmpq	40(%rsp), %rsi
	je	.L104
	movq	%rax, (%rsi)
	addq	$8, %rsi
	movq	%rsi, 32(%rsp)
.L105:
	decl	%r13d
	jne	.L106
	movq	32(%rsp), %r14
	movq	24(%rsp), %r12
	cmpq	%r12, %r14
	je	.L107
	movq	%r14, %r13
	subq	%r12, %r13
	movq	%r13, %rax
	sarq	$3, %rax
	movl	$63, %edx
	lzcntq	%rax, %rax
	subl	%eax, %edx
	movslq	%edx, %rdx
	addq	%rdx, %rdx
	movq	%r14, %rsi
	movq	%r12, %rdi
	call	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0
	leaq	8(%r12), %rbx
	cmpq	$128, %r13
	jle	.L108
	leaq	128(%r12), %r13
	jmp	.L114
.L156:
	cmpq	%rbx, %r12
	je	.L110
	movl	$8, %eax
	movq	%rbx, %rdx
	addq	%r12, %rax
	subq	%r12, %rdx
	movq	%r12, %rsi
	movq	%rax, %rdi
	call	memmove@PLT
.L110:
	movq	%r15, (%r12)
.L111:
	addq	$8, %rbx
	cmpq	%rbx, %r13
	je	.L155
.L114:
	movq	(%rbx), %r15
	cmpq	(%r12), %r15
	jb	.L156
	movq	-8(%rbx), %rdx
	leaq	-8(%rbx), %rax
	cmpq	%rdx, %r15
	jnb	.L131
.L113:
	movq	%rdx, 8(%rax)
	movq	%rax, %rsi
	subq	$8, %rax
	movq	(%rax), %rdx
	cmpq	%rdx, %r15
	jb	.L113
.L112:
	movq	%r15, (%rsi)
	jmp	.L111
.L104:
	leaq	8(%rsp), %rdx
	leaq	24(%rsp), %rdi
.LEHB0:
	call	_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
	jmp	.L105
.L155:
	cmpq	%r13, %r14
	je	.L107
.L115:
	movq	0(%r13), %rcx
	movq	-8(%r13), %rdx
	leaq	-8(%r13), %rax
	cmpq	%rdx, %rcx
	jnb	.L132
.L118:
	movq	%rdx, 8(%rax)
	movq	%rax, %rsi
	subq	$8, %rax
	movq	(%rax), %rdx
	cmpq	%rdx, %rcx
	jb	.L118
.L117:
	addq	$8, %r13
	movq	%rcx, (%rsi)
	cmpq	%r13, %r14
	jne	.L115
.L107:
	movq	24(%rsp), %rdx
	movq	32(%rsp), %rax
	leaq	_ZSt4cout(%rip), %rdi
	subq	%rdx, %rax
	sarq	$3, %rax
	shrq	%rax
	movq	(%rdx,%rax,8), %rbx
	movabsq	$1000000000000000, %rax
	xorl	%edx, %edx
	divq	%rbx
	movq	%rax, %rsi
	call	_ZNSo9_M_insertImEERSoT_@PLT
	movl	$6, %edx
	leaq	.LC1(%rip), %rsi
	movq	%rax, %rdi
	movq	%rax, %r12
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r12, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	movabsq	$4835703278458516699, %rdx
	movq	%rbx, %rax
	mulq	%rdx
	leaq	_ZSt4cout(%rip), %rdi
	movq	%rdx, %rsi
	shrq	$18, %rsi
	call	_ZNSo9_M_insertImEERSoT_@PLT
	movl	$6, %edx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movq	%rax, %r12
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r12, %rdi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE0:
	movq	24(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L127
	movq	40(%rsp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L127:
	movq	56(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L157
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	xorl	%eax, %eax
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L108:
	.cfi_restore_state
	movl	$8, %r15d
	cmpq	%rbx, %r14
	jne	.L120
	jmp	.L107
	.p2align 4,,10
	.p2align 3
.L158:
	cmpq	%rbx, %r12
	je	.L122
	movq	%rbx, %rdx
	leaq	(%r12,%r15), %rax
	subq	%r12, %rdx
	movq	%r12, %rsi
	movq	%rax, %rdi
	call	memmove@PLT
.L122:
	movq	%r13, (%r12)
.L123:
	addq	$8, %rbx
	cmpq	%rbx, %r14
	je	.L107
.L120:
	movq	(%rbx), %r13
	cmpq	(%r12), %r13
	jb	.L158
	movq	-8(%rbx), %rdx
	leaq	-8(%rbx), %rax
	cmpq	%r13, %rdx
	jbe	.L133
.L125:
	movq	%rdx, 8(%rax)
	movq	%rax, %rcx
	subq	$8, %rax
	movq	(%rax), %rdx
	cmpq	%rdx, %r13
	jb	.L125
.L124:
	movq	%r13, (%rcx)
	jmp	.L123
.L132:
	movq	%r13, %rsi
	jmp	.L117
.L131:
	movq	%rbx, %rsi
	jmp	.L112
.L133:
	movq	%rbx, %rcx
	jmp	.L124
.L157:
	call	__stack_chk_fail@PLT
.L134:
	endbr64
	movq	%rax, %r12
	jmp	.L128
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA7928:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE7928-.LLSDACSB7928
.LLSDACSB7928:
	.uleb128 .LEHB0-.LFB7928
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L134-.LFB7928
	.uleb128 0
.LLSDACSE7928:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC7928
	.type	main.cold, @function
main.cold:
.LFSB7928:
.L128:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	movq	24(%rsp), %rdi
	movq	40(%rsp), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L153
	vzeroupper
	call	_ZdlPvm@PLT
.L129:
	movq	%r12, %rdi
.LEHB1:
	call	_Unwind_Resume@PLT
.LEHE1:
.L153:
	vzeroupper
	jmp	.L129
	.cfi_endproc
.LFE7928:
	.section	.gcc_except_table
.LLSDAC7928:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC7928-.LLSDACSBC7928
.LLSDACSBC7928:
	.uleb128 .LEHB1-.LCOLDB3
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSEC7928:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE3:
	.section	.text.startup
.LHOTE3:
	.p2align 4
	.type	_GLOBAL__sub_I__Z13avx512_memcpyPvS_m, @function
_GLOBAL__sub_I__Z13avx512_memcpyPvS_m:
.LFB8588:
	.cfi_startproc
	endbr64
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	leaq	_ZStL8__ioinit(%rip), %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	__dso_handle(%rip), %rdx
	leaq	_ZStL8__ioinit(%rip), %rsi
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE8588:
	.size	_GLOBAL__sub_I__Z13avx512_memcpyPvS_m, .-_GLOBAL__sub_I__Z13avx512_memcpyPvS_m
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z13avx512_memcpyPvS_m
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 10.5.0-1ubuntu1~20.04) 10.5.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
