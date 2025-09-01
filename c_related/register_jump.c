#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void func1(unsigned long a, unsigned long  b) {
    const char *fmt = "a = %ld, b = %ld\n";
    const char *fmt2 = "c = 0x%lx, d = %ld\n";
    // assembly call printf with a and b as arguments
    asm volatile(
        "movq %0, %%rdi\n"
        "movq %2, %%rsi\n"
        "movq %3, %%rdx\n"
        "xorq %%rax, %%rax\n"
        "jmp label1\n"
        "call printf\n"
        "label1:\n"
        "movq %1, %%rdi\n"
        // "movq label1, %2\n"
        // "movq %2, %%rsi\n"
        "leaq label1(%%rip), %%rsi\n"
        // "movq %%eflag, %%rsi\n"
        "call printf"
        :
        : "r"(fmt), "r"(fmt2), "r"(a), "r"(b)
        : "rdi", "rsi", "rdx", "rax");
}


struct pt_regs {
/*
 * C ABI says these regs are callee-preserved. They aren't saved on kernel entry
 * unless syscall needs a complete, fully filled "struct pt_regs".
 */
	unsigned long r15;
	unsigned long r14;
	unsigned long r13;
	unsigned long r12;
	unsigned long bp;
	unsigned long bx;
/* These regs are callee-clobbered. Always saved on kernel entry. */
	unsigned long r11;
	unsigned long r10;
	unsigned long r9;
	unsigned long r8;
	unsigned long ax;
	unsigned long cx;
	unsigned long dx;
	unsigned long si;
	unsigned long di;
/*
 * On syscall entry, this is syscall#. On CPU exception, this is error code.
 * On hw interrupt, it's IRQ number:
 */
	unsigned long orig_ax;
/* Return frame for iretq */
	unsigned long ip;
	unsigned long cs;
	unsigned long flags;
	unsigned long sp;
	unsigned long ss;
/* top of stack page */
};

static struct pt_regs *skip_context = NULL;

unsigned long func2(void *dst, const void *src, unsigned long size) {
    unsigned long ret = -1;
    struct pt_regs *regs = skip_context + sizeof(struct pt_regs);  // move the pointer to end of before we push the regs
    // assembly call printf with c and d as arguments
    asm volatile(
        "movq %%ss, %%rdi\n\t"
        "movq %%rdi, -1*8(%1)\n\t"
        "movq %%rsp, -2*8(%1)\n\t"
        "movq %1, %%rsp\n\t"
        "subq $2*8, %%rsp\n\t"
        "pushfq\n\t"
        "push $-1\n\t"                // reserve a space for cs, because we cannot directly push cs
        "leaq skip_copy_to_user(%%rip), %%rdi\n\t"
        "pushq %%rdi\n\t"
        "pushq $-1\n\t"               // we don't have error code
        "pushq %%rdi\n\t"             // has nothing
        "pushq %%rsi\n\t"
        "pushq %%rdx\n\t"
        "pushq %%rcx\n\t"
        "pushq %%rax\n\t"
        "pushq %%r8\n\t"
        "pushq %%r9\n\t"
        "pushq %%r10\n\t"
        "pushq %%r11\n\t"
        "pushq %%rbx\n\t"
        "pushq %%rbp\n\t"
        "pushq %%r12\n\t"
        "pushq %%r13\n\t"
        "pushq %%r14\n\t"
        "pushq %%r15\n\t"
        "movq %%cs, %%rdi\n\t"           // move cs to rdi, we will use it anyway
        "movq %%rdi, -4*8(%1)\n\t"       // save the cs
        "movq -2*8(%1), %%rsp\n\t"      // restore the stack pointer
        "movq %2, %%rdi\n\t"          // 1st argument: dst
        "movq %3, %%rsi\n\t"          // 2nd argument: src
        "movq %4, %%rdx\n\t"          // 3rd argument: size
        "xorq %%rax, %%rax\n\t"
        "call memcpy\n\t"
        "movq %%rax, %0\n\t"
        "skip_copy_to_user:\n\t"
    : "=r"(ret)
    : "r"(regs), "m"(dst), "m"(src), "m"(size)
    : "rdi", "rsi", "rdx", "rax", "memory");

    memset(skip_context, 0, sizeof(struct pt_regs));
    return ret;
}

int foo;
void *addr_inside_function() {
    foo++;

    lab1:  ;  // labels only go on statements, not declarations
    void *tmp = &&lab1;

    foo++;
    return tmp;
}


int main() {
    unsigned long a = 5;
    unsigned long b = 10;
    const char *text1 = "Hello, World!";
    const size_t size = strlen(text1) + 1;
    const char *text2;

    skip_context = (struct pt_regs *)malloc(sizeof(struct pt_regs));
    text2 = (const char *)malloc(size);

    func2((void *)text2, (void *)text1, size);
    // func1(a, b);
    // void *k = addr_inside_function();
    // printf("k = %p\n", k);
    free((void *)text2);
    free(skip_context);
    return 0;
}