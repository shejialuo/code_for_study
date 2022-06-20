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
size is 4096 bits, which means $2^{12}$, so for the start address of one
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

### Translate virtual address to physical address

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

First, it defines a pointer to `pte_t`, which is defined in `riscv.h`,
just a type alias: `typedef uint64 pte_t`. `MAXVA` is also defined in `riscv.h`:
`#define MAXVA (1L << (9 + 9 + 9 + 12 - 1))`. Apparently, `MAXVA` is
actually one bit less than the max allowed by Sv39, to avoid having
to sign-extend virtual addresses that have the high bit set. Next step, we want to get the `pte` from the `walk` function.

We look at the `walk` function.

```c
pte_t *walk(pagetable_t pagetable, uint64 va, int alloc) {
  if(va >= MAXVA)
    panic("walk);
  for(int level = 2; level > 0; level--) {
    pte_t *pte = &pagetable[PX(level, va)];
    if(*pte & PTE_V) {
      pagetable = (pagetable_T)PTE2PA(*pte);
    } else {
      if(!alloc || (pagetable = (pde_t*)kalloc()) == 0)
        return 0;
      memset(pagetable, 0, PGSIZE);
      *pte = PA2PTE(pagetable) | PTE_V;
    }
  }
  return &pagetable[PX(0, va)];
}
```

Well, `pagetable_t` is a type alias defined in `riscv.h`: `typedef uint64 *pagetable_t`.
For Sv39, we have three `pagetable_t` whose size is 512. So, at first,
we check the virtual address. As mentioned above, Sv39 has three levels
of page-table pages. So we should use `for` to do this operation.

We need to get the entry offset from the `va`, so in `rsicv.h`, xv6 defines
the following macros:

```c
#define PXMASK 0x1FF // 9bits
#define PXSHIFG(level) (PGSHIFT+(9*(level)))
#define PX(level, va) ((((uint64) (va)) >> PXSHIFT(level)) & PXMASK)
```

It is easy to understand the above code, for virtual address we should
first right shift 12 bits and for level 0, right shift 0 \* 9 bits,
for level 1, right shift 1 \* 9 bits and so on.

Now we get the level 2, we can find the pte, if it is valid, we
should use `PTE2PA()` macro to convert the pte to the address for get
level 1's address

```c
#define PTE2PA(pte) (((pte) >> 10) << 12)
```

As we have talked about, the lower 10 bit is flags, so first we should
right shift 10 bits of the pte, and right shift 12 bits. This is an iterative operation,
so the assign it to the `pagetable`.

However, what if the pte is not valid. So in `walk`, we offer `alloc` flag,
if allocating the memory is allowed, we then allocate the memory using `kalloc()`.
Here, we could get an important information: when the
directory doesn't exist,
xv6 allocates the memory for the directory, which is just one page memory.
We could do the calculation. Because the `pagetable_t` could
accommodate 512 pte, and each pte is 8 bytes. so 8 \* 512 = 4096 bytes.
When we finish the memory allocation, we need to make the pte to
be equal to the `pagetable`. And we need to use `PA2PTE` in `riscv.h` which
is the revertible operation of the `PTE2PA` to set the value of pte.

```c
((((uint64)pa) >> 12) << 10)
```

At last, we could get the physical address's top 44 bits.

After we get the `pte`, we make sure the `pte` is valid and
not user. Then we use `PTE2PA` to get the physical address. Thus we can translate
virtual address to physical address.

### Map physical address to virtual address

#### Memory Layout

Well, before we dive into mapping physical address to virtual address,
we first look at how xv6 organizes the virtual address.

`memlayout.h` defines the layout. And there are many details about the qemu
spec, too tedious.

The code is easy to understand, I omit detail here.

#### Kernel

xv6 first defines the kernel's page table.

```c
pagetable_t kernel_pagetable
```

xv6 uses `kvminit` to initialize the kernel's page table, it is
just a wrapper function.

```c
void kvminit(void) {
  kernel_pagetable = kvmmake();
}
```

`kvvmake` actually makes a direct-map page table for the kernel.

```c
  pagetable_t kpgtbl;

  kpgtbl = (pagetable_t) kalloc();
  memset(kpgtbl, 0, PGSIZE);

  // uart registers
  kvmmap(kpgtbl, UART0, UART0, PGSIZE, PTE_R | PTE_W);

  // virtio mmio disk interface
  kvmmap(kpgtbl, VIRTIO0, VIRTIO0, PGSIZE, PTE_R | PTE_W);

  // PLIC
  kvmmap(kpgtbl, PLIC, PLIC, 0x400000, PTE_R | PTE_W);

  // map kernel text executable and read-only.
  kvmmap(kpgtbl, KERNBASE, KERNBASE, (uint64)etext-KERNBASE, PTE_R | PTE_X);

  // map kernel data and the physical RAM we'll make use of.
  kvmmap(kpgtbl, (uint64)etext, (uint64)etext, PHYSTOP-(uint64)etext, PTE_R | PTE_W);

  // map the trampoline for trap entry/exit to
  // the highest virtual address in the kernel.
  kvmmap(kpgtbl, TRAMPOLINE, (uint64)trampoline, PGSIZE, PTE_R | PTE_X);

  // map kernel stacks
  proc_mapstacks(kpgtbl);

  return kpgtbl;
```

Now, we need to see `kvvmmap`, well, it is also a wrapper function.

```c
void kvmmap(pagetable_t kpgtbl, uint64 va, uint64 pa, uint64 sz, int perm) {
  if(mappages(kpgtbl, va, sz, pa, perm) != 0)
    panic("kvmmap");
}
```

Let's see `mappages`, the core part.

```c
int mappages(pagetable_t pagetable, uint64 va, uint64 size, uint64 pa, int perm) {
  uint64 a, last;
  pte_t *pte;

  if(size == 0)
    panic("mappages: size");

  a = PGROUNDDOWN(va);
  last = PGROUNDDOWN(va + size - 1);
  for(;;) {
    if((pte = walk(pagetable, a, 1)) == 0)
      return -1;
    if(*pte & PTE_V)
      panic("mappages: remap");
    *pte = PA2PTE(pa) | perm | PTE_V;
    if(a == last)
      break;
    a += PGSIZE;
    pa += PGSIZE;
  }
  return 0;
}
```

It is super easy! Because the detail is hidden in the `walk` which we
have talked about.

#### User

For user, the `uvminit` loads the user initcode into address 0 of
pagetable for every first process.

```c
void uvminit(pagetable_t pagetable, uchar *src, uint sz) {
  char *mem;

  if(sz >= PGSIZE)
    panic("inituvm: more than a page")
  mem = kalloc();
  memset(mem, 0, PGSIZE);
  mappages(pagetable, 0, PGSIZE, (uint64)mem, PTE_W|PTE_R|PTE_X|PTE_U);
  memmove(mem, src, sz);
}
```

For each process, we need to create a page table. This is what `uvmcreate` does.

```c
pagetable_t uvmreate() {
  pagetable_t pagetable;
  pagetable = (pagetable_t)kalloc();
  if(pagetable == 0)
    return 0;
  memset(pagetable, 0, PGSIZE);
  return pagetable;
}
```

We have known that `mappages` can map physical address to virtual address,
xv6 defines `uvmunmap` to unmap and optionally free the physical memory.

```c
void uvmunmap(pagetable_t pagetable, uint64 va, uint64 npages, int do_free) {
  uint64 a;
  pte_t *pte;

  if((va % PG_SIZE) != 0)
    panic("uvmunmap: not aligned");

  for(a = va; a < va + npages*PGSIZE; a += PGSIZE) {
    if((pte = walk(pagetable, a, 0)) == 0)
      panic("uvmunmap: walk");
    if((*pte & PTE_V) == 0)
      panic("uvmunmap: not mapped");
    if(PTE_FLAGS(*pte) == PTE_V)
      panic("uvmunmap: not a leaf");
    if(do_free){
      uint64 pa = PTE2PA(*pte);
      kfree((void*)pa);
    }
    *pte = 0;
  }
}
```

Well, as you can see, the most important thing is to find how virtual memory
is associated with physical memory. The other functions now are easy to understand.
I omit detail here.
