# Kernel memory

The virtual memory is essential. The first step is to understand
what functionality the hardware provides.

## Paging hardware

xv6 runs on Sv39 RISC-V, which means that only the bottom 39 bits
of a 64-bit virtual address are used; the top 25 bits are not used.
In this Sv39 configuration, a RISC-V page is logically an array
of $2^{27}$ *page table entries*(PTEs). Each PTE contains a 44-bit physical
page number(PNN) and some flags.

The paging hardware translates a virtual address by using the top 27
bits of the 39 bits to index into the page table to find a PTE,
and making a 56-bit physical address whose top 44 bits come from the PPN in PTE,
and whose bottom 12 bits are copied from the original virtual address.
A page table gives the operating system control over virtual-to-physical
address translations at the granularity of aligned chunks of
4096 bytes. Such a chunk is called a *page*.

![RISC-V virtual anb physical addresses](https://s2.loli.net/2022/06/06/tULIYEdGh3j9F1o.png)

The three-level structure allows a memory-efficient in which large
ranges of virtual addresses have no mappings, the three-level
can omit entire page directories.

![RISC-V address translation details](https://s2.loli.net/2022/06/06/pFzaiEU7LS2ZdAX.png)

Each PTE contains flag bits that tell the paging hardware how
the associated virtual address is allowed to be used.

So in `riscv.h`, the code defines the following attributes:

```c
#define PGSZIE 4096
#define PGSHIFT 12

#define PTE_V (1L << 0)
#define PTE_R (1L << 1)
#define PTE_W (1L << 2)
#define PTE_X (1L << 3)
#define PTE_U (1L << 4)
```

To tell the hardware to use a page table, the kernel must write
the physical address of the root page-table page into the `satp` register.
Each CPU has its own `satp`. A CPU will translate all addresses generated
by subsequent instructions using the page table pointed to by its
own `satp`. Each CPU has its own `satp` so that different CPUs can
run different processes, each with a private address space described
by its own page table.

## Memory management

In order to achieve virtual memory functionality, the kernel should first
do memory allocation and maintain a table(whatever) to record the unused page.
Well, the common way is to use a bitmap. However, xv6 uses list.

First, xv6 constructs a basic abstract list:

```c
struct run {
  struct run *next;
}
```

Next, it defines an unnamed `struct` which hold a spin lock and a list node:

```c
struct {
  struct spinlock lock;
  struct run* freelist;
}
```

In `memlayout.h`, the code defines the physical start address and end address:

```c
#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 128*1024*1024)
```

However, we can't just divide the memory interval `[KEENBASE, PHYSTOP]`.
Because, we should preserve some address for kernel itself, so in
`kernel.ld`, we define the address `end` for the physical start address.
And we declare it in the `kalloc.c`

```c
extern char end[];
```

Now, we should initialization the memory, first it should initialize
the spin lock and split the memory into fixed 4096 bytes one page.

```c
void kinit() {
  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}
```

Here, we use `freerange` to wrap:

```c
void freerange(void *pa_start, void *pa_end) {
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}
```

This function is easy. First, it uses `PGROUNDUP` macro to find
the *next* start address of the page. This macro is defined in the `riscv.h`,
also it also defines `PGROUNDDOWN` macro, which finds the start address of the page
which incorporates the `pa_start`.

```c
#define PGROUNDUP(sz) (((sz)+PGSIZE-1) & ~(PGSIZE-1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE-1))
```

It may seem difficult at fist glance. But the idea is simple. Because the page
size is 4096 bits, which means $2^12$, so for the start address of one
page, it must be `0x__000000000000`. So for the UP operation,
we first add the `sz` and make the lower 12 bits to 0, and keep the upper bits the same.
For the Down operation, we just make the lower 12 bits to 0, and keep the upper bits the same.

The most important part is `kfree` function, it may seem confusing, the physical
memory is there, why free? Well, actually we don't free the memory,
what we want to do is maintain the free list. This is the key.

```c
void kfree(void *pa) {
  struct run *r;
  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree);

  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;
  acquire(&kmem.lock);
  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock)
}
```

Well, the function first do the check:

+ Is the address is the `0x__000000000000`?
+ Is the address is below the `end`?
+ Is the address is upper than `PHYSTOP`?

If the check failed, it will invoke the `panic` function defined in `print.c`.
And use `memset` to set the junk content.

The interesting part is it converts the `pa` to `struct run*`. Now,
we can infer that for every `struct run*` type, its value is one of
the start address of a page. And use head insert to form a free page
list where `kmem.freelist` always points to the head.

Last, we want to request a page to use, how should we do this? We should
find the current free list head and make the head to point to the next free list node.

```c
void * kalloc(void) {
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  if(r)
    kmem.freelist = r->next;

  if(r)
    memset((char*)r, 5, PGSIZE); // fill with junk
  return (void*)r;
}
```

Thus, in `main.c`, it calls `kinit` to do physical page allocation.

## Paging software

Now that we have split the physical memory to fixed-size pages, we
need to do more work for virtual memory. In this moment, the `kalloc` function
returns the actual physical address. For kernel or user, the address is virtual.
So we should at least do two jobs:

1. Write code to translate virtual address to physical address
2. Write code to map physical address to virtual address.

The risc-v Sv39 scheme has three levels of page-table pages. A page-table page
contains 512 64-bit PTEs. A 64-bit virtual address is split into
five fields:

+ 39..63 -- must be zero.
+ 30..38 -- 9 bits of level-2 index.
+ 21..29 -- 9 bits of level-1 index.
+ 12..20 -- 9 bits of level-0 index.
+ 0..11  -- 12 bits of byte offset within the page.

The xv6 defines `walkaddr` to translate a virtual address to physical address.

```c
uint64 walkaddr(pagetable_t pagetable, uint64 va) {
  pte_t *pte;
  uint64 pa;

  if(va >= MAXVA)
    return 0;

  pte = walk(pagetable, va, 0);
  if(pte == 0)
    return 0;
  if((*pte) & PTE_V == 0)
    return 0;
  if((*pte) & PTE_U == 0)
    return 0;
  pa = PTE2PA(*pte);
  return pa;
}
```
