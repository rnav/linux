// SPDX-License-Identifier: GPL-2.0-only
#include <linux/module.h>

#include <linux/mm.h> /* for handle_mm_fault() */
#include <linux/ftrace.h>
#include <linux/kprobes.h>

void my_direct_func(struct vm_area_struct *vma,
			unsigned long address, unsigned int flags)
{
	trace_printk("handle mm fault vma=%p address=%lx flags=%x\n",
		     vma, address, flags);
}

extern void my_tramp(void *);

static unsigned long my_ip = (unsigned long)handle_mm_fault;

#ifdef CONFIG_X86
asm (
"	.pushsection    .text, \"ax\", @progbits\n"
"	.type		my_tramp, @function\n"
"   my_tramp:"
"	pushq %rbp\n"
"	movq %rsp, %rbp\n"
"	pushq %rdi\n"
"	pushq %rsi\n"
"	pushq %rdx\n"
"	call my_direct_func\n"
"	popq %rdx\n"
"	popq %rsi\n"
"	popq %rdi\n"
"	leave\n"
"	ret\n"
"	.size		my_tramp, .-my_tramp\n"
"	.popsection\n"
);
#elif CONFIG_PPC64
asm (
"	.pushsection	.text, \"ax\", @progbits\n"
"	.type		my_tramp, @function\n"
"	.global		my_tramp\n"
"   my_tramp:\n"
"	std	0, 16(1)\n"
"	stdu	1, -480(1)\n"
"	std	2, 24(1)\n"
"	std	3, 136(1)\n"
"	std	4, 144(1)\n"
"	std	5, 152(1)\n"
"	mflr	7\n"
"	std	7, 368(1)\n"
"	bcl	20, 31, 1f\n"
"1:	mflr	12\n"
"	ld	2, (2f - 1b)(12)\n"
"	bl	my_direct_func\n"
"	nop\n"
"	ld	5, 152(1)\n"
"	ld	4, 144(1)\n"
"	ld	3, 136(1)\n"
"	ld	2, 24(1)\n"
"	ld	7, 368(1)\n"
"	mtctr	7\n"
"	addi	1, 1, 480\n"
"	ld	0, 16(1)\n"
"	mtlr	0\n"
"	bctr\n"
"	.size		my_tramp, .-my_tramp\n"
"2:\n"
"	.quad		.TOC.@tocbase\n"
"	.popsection\n"
);
#endif


static int __init ftrace_direct_init(void)
{
#ifdef CONFIG_PPC64
	my_ip = ppc_function_entry((void *)my_ip) + 4;
#endif
	return register_ftrace_direct(my_ip, (unsigned long)my_tramp);
}

static void __exit ftrace_direct_exit(void)
{
	unregister_ftrace_direct(my_ip, (unsigned long)my_tramp);
}

module_init(ftrace_direct_init);
module_exit(ftrace_direct_exit);

MODULE_AUTHOR("Steven Rostedt");
MODULE_DESCRIPTION("Another example use case of using register_ftrace_direct()");
MODULE_LICENSE("GPL");
