#include <asm/ppc_asm.h>

/* do_mfspr(sprn, default_value) */
_GLOBAL(do_mfspr)
	mflr	r0
	bcl	20, 31, 1f		/* get address of next instr into LR */
1:	rlwinm	r3, r3, 3, 0x3ff * 8	/* sprn = (sprn & 0x3ff) << 3 */
	mflr	r5
	mtlr	r0
	add	r5, r5, r3
	addi	r5, r5, 2f - 1b
	mtctr	r5
	mr	r3, r4
	bctr
2:
	spr = 0
	.rept	1024
	mfspr	r3, spr
	blr
	spr = spr + 1
	.endr

/* do_mtspr(sprn, new_value) */
_GLOBAL(do_mtspr)
	mflr	r0
	bcl	20, 31, 1f		/* get address of next instr into LR */
1:	rlwinm	r3, r3, 3, 0x3ff * 8	/* sprn = (sprn & 0x3ff) << 3 */
	mflr	r5
	mtlr	r0
	add	r5, r5, r3
	addi	r5, r5, 2f - 1b
	mtctr	r5
	bctr
2:
	spr = 0
	.rept	1024
	mtspr	spr, r4
	blr
	spr = spr + 1
	.endr
