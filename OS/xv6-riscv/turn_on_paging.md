# Turn On Paging

We have already talked about how physical memory are paged and
virtual memory is mapped. Now we need to turn on paging for RISC-V.

In `vm.c`, the kernel uses `kvminithart` to turn on paging functionality.

```c
void kvminithart() {
  w_satp(MAKE_SATP(kernel_pagetable));
  sfence_vma();
}
```

In RISC-V, `satp` is the supervisor address translation and protection
register, which means we should configure the `satp` for paging. You could
see the RISC-V for detail.

In `riscv.h`, the xv6 wraps the assembly code:

```c
static inline void w_satp(uint64 x) {
  asm volatile("csrw satp, %0" : : "r" (x));
}

static inline uint64 r_satp() {
  uint64 x;
  asm volatile("csrr %0, satp" : "=r" (x) );
  return x;
}
```

Because we use SV39, so we should write the corresponding value:

```c
#define SATP_SV39 (8L << 60)

#define MAKE_SATP(pagetable) (SATP_SV39 | (((uint64)pagetable) >> 12))
```

At last, we should flush the TLB.

```c
static inline void sfence_vma() {
  asm volatile("sfence.vma zero, zero");
}
```
