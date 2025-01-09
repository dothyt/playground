#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/uaccess.h>
#include <asm/asm.h>
#include <asm/errno.h>

void blatant_div_by_zero(void)
{
        int q, d;

        d = 0;
        asm volatile ("movl $20, %%eax;"
                "movl $0, %%edx;"
                "1: div %1;"
                "movl %%eax, %0;"
                "2:\t\n"
                "\t.section .fixup,\"ax\"\n"
                "3:\tmov\t$-1, %0\n"
                "\tjmp\t2b\n"
                "\t.previous\n"
                _ASM_EXTABLE(1b, 3b)
                : "=r"(q)
                : "b"(d)
                :"%eax"
        );

        pr_info("q = %d\n", q);
}

// static inline int futex_atomic_cmpxchg_inatomic(u32 *uval, u32 __user *uaddr,
// 						u32 oldval, u32 newval)
// {
// 	int ret = 0;

// 	if (!user_access_begin(uaddr, sizeof(u32)))
// 		return -EFAULT;
// 	asm volatile("\n"
// 		"1:\t" LOCK_PREFIX "cmpxchgl %4, %2\n"
// 		"2:\n"
// 		"\t.section .fixup, \"ax\"\n"
// 		"3:\tmov     %3, %0\n"
// 		"\tjmp     2b\n"
// 		"\t.previous\n"
// 		_ASM_EXTABLE_UA(1b, 3b)
// 		: "+r" (ret), "=a" (oldval), "+m" (*uaddr)
// 		: "i" (-EFAULT), "r" (newval), "1" (oldval)
// 		: "memory"
// 	);
// 	user_access_end();
// 	*uval = oldval;
// 	return ret;
// }

static inline int test_and_set_bit_user(bool *uval, u64 __user *uaddr, long nr)
{
	int ret = 0;
    bool old_bit;

    /* `c` is the Carry flag set after bts */
	if (!user_access_begin(uaddr, sizeof(u64)))
		return -EFAULT;
	asm volatile("\n"
		"1:\t" LOCK_PREFIX "btsq %[val], %[var]" CC_SET(c)
		"2:\n"
		"\t.section .fixup, \"ax\"\n"
		"3:\tmov     %3, %0\n"
		"\tjmp     2b\n"
		"\t.previous\n"
		_ASM_EXTABLE_UA(1b, 3b)
		: "+r" (ret), [var] "+m" (*uaddr), CC_OUT(c) (old_bit)
		: "i" (-EFAULT), [val] "Ir" (nr)
		: "memory"
	);
	user_access_end();
	*uval = old_bit;
	return ret;
}

static inline int test_and_clear_bit_user(bool *uval, u64 __user *uaddr, long nr)
{
	int ret = 0;
    bool old_bit;

    /* `c` is the Carry flag set after bts */
	if (!user_access_begin(uaddr, sizeof(u64)))
		return -EFAULT;
	asm volatile("\n"
		"1:\t" LOCK_PREFIX "btrq %[val], %[var]" CC_SET(c)
		"2:\n"
		"\t.section .fixup, \"ax\"\n"
		"3:\tmov     %3, %0\n"
		"\tjmp     2b\n"
		"\t.previous\n"
		_ASM_EXTABLE_UA(1b, 3b)
		: "+r" (ret), [var] "+m" (*uaddr), CC_OUT(c) (old_bit)
		: "i" (-EFAULT), [val] "Ir" (nr)
		: "memory"
	);
	user_access_end();
	*uval = old_bit;
	return ret;
}

static int __init fixup_init(void)
{
    bool test;
    unsigned long test_addr[2] = {0, 0};
    long nr;

    pr_info("fixup_init\n");

    for (nr = 0; nr < 64; nr++) {
        test_and_set_bit_user(&test, (u64 __user *)&test_addr, nr);
        pr_info("nr: %ld test_addr[0] = %lx, test_addr[1] = %lx\n", nr, test_addr[0], test_addr[1]);
    }

    for (nr = 0; nr < 64; nr++) {
        test_and_set_bit_user(&test, (u64 __user *)&test_addr[1], nr);
        pr_info("nr: %ld test_addr[0] = %lx, test_addr[1] = %lx\n", nr, test_addr[0], test_addr[1]);
    }

    pr_info("clearing bits\n");

    for (nr = 0; nr < 64; nr++) {
        test_and_clear_bit_user(&test, (u64 __user *)&test_addr, nr);
        pr_info("nr: %ld test_addr[0] = %lx, test_addr[1] = %lx\n", nr, test_addr[0], test_addr[1]);
    }

    for (nr = 0; nr < 64; nr++) {
        test_and_clear_bit_user(&test, (u64 __user *)&test_addr[1], nr);
        pr_info("nr: %ld test_addr[0] = %lx, test_addr[1] = %lx\n", nr, test_addr[0], test_addr[1]);
    }
    return 0;
}

static void __exit fixup_exit(void)
{
    pr_info("fixup_exit\n");
}

module_init(fixup_init);
module_exit(fixup_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Atomic Operation in userspace by kernel module");
