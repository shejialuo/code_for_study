# Kernel Start

For Qemu, the kernel start address is `0x00008000`, so the `kernel.ld` code defines
the entry symbol is `_entry` which is defined in `entry.S` file.

So the kernel actually starts from `entry.S`:

```assembly
_entry:
  la sp, stack0
  li a0, 1024*4
  csrr a1, mhartid
  addi a1, a1, 1
  mul a0, a0, a1
  add sp, sp, a0
  call start
spin:
  j spin
```

In xv6, it allows multiple CPUs, so for per CPU, we need one stack.
So firstly, we use `la` to get the stack start address symbol `stack0`.
But we should actually put the `sp` to the stack top, so we need to find the
number of CPUs.
So we get the information from the `mhartid` register by `csrr a1, mhartid`.
Then just use random access to set the `sp` and call the function `start`
in `start.c`.

> The `start` function start performs some configuration that is only allowed in machine mode, and
> then switches to supervisor mode. To enter supervisor mode, RISC-V provides the instruction
> `mret`. This instruction is most often used to return from a previous call from supervisor mode to
> machine mode. `start` isn't returning from such a call, and instead sets things up as if there had
> been one: it sets the previous privilege mode to supervisor in the register `mstatus`, it sets the
> return address to `main` by writing `main`'s address into the register `mepc`,
> disables virtual address translation in supervisor mode by writing 0
> into the page-table register satp, and delegates all interrupts and exceptions to supervisor mode.
> Before jumping into supervisor mode, start performs one more task: it programs the clock
> chip to generate timer interrupts. With this housekeeping out of the way, `start`
> "returns" to supervisor mode by calling `mret`. This causes the program counter to change to `main`

After `main` initializes several devices and subsystems, it creates
the first process by calling `userinit`.

```c
void
main()
{
  if(cpuid() == 0){
    consoleinit();
    printfinit();
    printf("\n");
    printf("xv6 kernel is booting\n");
    printf("\n");
    kinit();         // physical page allocator
    kvminit();       // create kernel page table
    kvminithart();   // turn on paging
    procinit();      // process table
    trapinit();      // trap vectors
    trapinithart();  // install kernel trap vector
    plicinit();      // set up interrupt controller
    plicinithart();  // ask PLIC for device interrupts
    binit();         // buffer cache
    iinit();         // inode table
    fileinit();      // file table
    virtio_disk_init(); // emulated hard disk
    userinit();      // first user process
    __sync_synchronize();
    started = 1;
  } else {
    while(started == 0)
      ;
    __sync_synchronize();
    printf("hart %d starting\n", cpuid());
    kvminithart();    // turn on paging
    trapinithart();   // install kernel trap vector
    plicinithart();   // ask PLIC for device interrupts
  }

  scheduler();
}
```

The first process executes a small program written in RISC-V assembly:

```c
uvminit(p->pagetable, initcode, sizeof(initcode));
```

In the assembly code `initCode.s`, it uses the system call `exec` to
start a new process `init.c`, which initializes the console, thus
the OS is started.
