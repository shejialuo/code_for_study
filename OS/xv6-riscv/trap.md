# Trap

There are three kinds of event which cause the CPU to set aside
ordinary execution of instructions and force a transfer of
control to special code that handles the event. One situation is a system call,
when a user program executes the `ecall` instruction to ask the kernel
to do something for it. Another situation is an *exception*:
an instruction does some illegal. The third situation is a device *interrupt*.

xv6 uses *trap* as a generic term for these situations. xv6
handles all traps in the kernel. And xv6 trap handling proceeds
in four stages: hardware actions taken by the RISC-V CPU, some
assembly instructions that prepare the way for kernel C code,
a C function decides what to do with the trap, and the system call
or device-driver service routine.

## Hardware Support

In order to handle the trap, we need the trap vector. RISC-V supports
`stvec` register that holds trap vector configuration, consisting of
a vector base address (BASE) and a vector mode (MODE).You could see the RISC-V spec,
I omit detail here. In `riscv.h`, xv6 wraps the assembly code.

```c
static inline void w_stvec(uint64 x) {
  asm volatile("csrw stvec, %0" : : "r" (x));
}

static inline uint64 r_stvec() {
  uint64 x;
  asm volatile("csrr %0, stvec" : "=r" (x) );
  return x;
}
```

xv6 writes the address of its trap handler here; the RISC-V jumps to
the address in `stvec` to handle a trap. In `trap.c`, xv6 uses
`trapinithart` to write the address of the trap handler.

```c
void trapinithart(void) {
  w_stvec((uint64)kernelvec);
}
```

When a trap occurred, RISC-V saves the program counter in `sepc`.
The `srte` instruction copies `sepc` to the `pc`. The kernel can write `sepc` to control
where `sret` goes. Also, in `riscv.h`, the xv6 wraps the assembly code.

```c
static inline void w_sepc(uint64 x) {
  asm volatile("csrw sepc, %0" : : "r" (x));
}

static inline uint64 r_sepc() {
  uint64 x;
  asm volatile("csrr %0, sepc" : "=r" (x) );
  return x;
}
```

RISC-V puts a number in `scause` which describes the reason for the
trap. In `riscv.h`, the xv6 wraps the assembly code.

```c
static inline uint64 r_scause() {
  uint64 x;
  asm volatile("csrr %0, scause" : "=r" (x) );
  return x;
}
```

The kernel places a value in `sscratch` register at the
very start of a trap handler. In `riscv.h`, the xv6 wraps the assembly code.

```c
static inline void w_sscratch(uint64 x) {
  asm volatile("csrw sscratch, %0" : : "r" (x));
}
```

The SIE bit in `sstatus` register controls whether device interrupts
are enabled. If the kernel clears SIE, the RISC-V will defer device
interrupts until the kernel sets SIE. The SPP bit indicates whether a
trap came from user mode, and controls to what mode `sret` returns.

When it needs to force a trap, the RISC-V hardware does the following
for all trap types (other than timer interrupts):

1. If the trap is a device interrupt, and the `sstatus` SIE bit
is clear, don't do any of the following.
2. Disable interrupts by clearing the SIE bit in `sstatus`.
3. Copy the `pc` to `sepc`.
4. Save the current mode in the SPP bit in `sstatus`.
5. Set `scause` to reflect the trap's cause.
6. Set the mode to supervisor.
7. Copy `stvec` to the `pc`.
8. Start executing at the new `pc`.

## Kernel Support

### Trap From User Space

In `process_data_structure.md`, we have talked about what is the data
structure of a process, we have omitted `struct trapframe *trapframe`. Now,
let's talk about it.

When we handle traps from user to the kernel. The hardware just does
the basic thing for us, in order to keep the context, we need
to store the registers of the current context. So in the PCB, we have
stored a `trapframe` to
hold the context when handling traps. In `proc.h`, xv6 defines this structure.

```c
struct trapframe {
  /*   0 */ uint64 kernel_satp;   // kernel page table
  /*   8 */ uint64 kernel_sp;     // top of process's kernel stack
  /*  16 */ uint64 kernel_trap;   // usertrap()
  /*  24 */ uint64 epc;           // saved user program counter
  /*  32 */ uint64 kernel_hartid; // saved kernel tp
  /*  40 */ uint64 ra;
  /*  48 */ uint64 sp;
  /*  56 */ uint64 gp;
  /*  64 */ uint64 tp;
  /*  72 */ uint64 t0;
  /*  80 */ uint64 t1;
  /*  88 */ uint64 t2;
  /*  96 */ uint64 s0;
  /* 104 */ uint64 s1;
  /* 112 */ uint64 a0;
  /* 120 */ uint64 a1;
  /* 128 */ uint64 a2;
  /* 136 */ uint64 a3;
  /* 144 */ uint64 a4;
  /* 152 */ uint64 a5;
  /* 160 */ uint64 a6;
  /* 168 */ uint64 a7;
  /* 176 */ uint64 s2;
  /* 184 */ uint64 s3;
  /* 192 */ uint64 s4;
  /* 200 */ uint64 s5;
  /* 208 */ uint64 s6;
  /* 216 */ uint64 s7;
  /* 224 */ uint64 s8;
  /* 232 */ uint64 s9;
  /* 240 */ uint64 s10;
  /* 248 */ uint64 s11;
  /* 256 */ uint64 t3;
  /* 264 */ uint64 t4;
  /* 272 */ uint64 t5;
  /* 280 */ uint64 t6;
}
```

`trap.c` sets `setvc` to point to the `uservec` in the `trampoline.S`. So
traps from user space start from `uservec` and also makes `sscratch` points
to process's `p-trampframe`.

The assembly code is easy to understand. `uservec` keeps the context and
`userret` restores the context.

But what does the context the `uservec` keep? This is the point.

+ Store basic register of the users.
+ Change `sp` to be the kernel stack pointer.
+ Load the address of `usertrap()`
+ Change the current page table to the kernel page table.
+ Jump to the `usertrap()`.

Now let's look at `usertrap` in `trap.c`.

```c
void usertrap(void) {
  int which_dev = 0;
  if((r_sstatus() & SSTATUS_SPP) != 0)
    panic("usertrap: not from user mode");
  w_stvec((uint64)kernelvec);

  struct proc *p = myproc();
  p->tranpframe->epc = r_sepc();

  if(r_scayse() = 8) {
    // system call
    if(p->killed)
      exit(-1);
    p->tramframe->epc += 4;

    intr_on();
    syscall();
  } else if((which_dev = devintr()) != 0) {
    //ok
  } else {
    printf("usertrap(): unexpected scause %p pid=%d\n", r_scause(), p->pid);
    printf("            sepc=%p stval=%p\n", r_sepc(), r_stval());
    p->killed = 1;
  }

  if(p->killed)
    exit(-1);

  // give up the CPU if this is a timer interrupt.
  if(which_dev == 2)
    yield();

  usertrapret();
}
```

First, the code decides whether the trap comes from the user.
The `SSTATUS_SPP` is define in `riscv.h`.

```c
// Previous mode, 1=Supervisor, 0=User
#define SSTATUS_SPP (1L << 8)
```

And next xv6 uses `w_stvec` to write the trap handle address
to `stvec`. Remember, when trap occurs, RISC-V saves the original `pc`
to `sepc`, so we use `p->trapframe->epc = r_sepc()` to save user
program counter. The remain is easy to understand. I omit detail here.

For `usertrapret`, it is just the same as the above. However, we need
to store the kernel's page table and some other thing in order
to re-enter the kernel.

### Trap From Kernel

Since we're already in the kernel. We just first save all
the registers and call `kerneltrap` in `trap.c` and
restores the register. This is what `kernelvec` in `kernelvec.S` does.

Let's look at the `kerneltrap` in `trap.c`:

```c
void kerneltrap() {
  int which_dev = 0;
  uint64 sepc = r_sepc();
  uint64 sstatus = r_sstatus();
  uint64 scause = r_scause();

  if((sstatus & SSTATUS_SPP) == 0)
    panic("kerneltrap: not from supervisor mode");
  if(intr_get() != 0)
    panic("kerneltrap: interrupts enabled");
  if((which_dev = devintr()) == 0){
    printf("scause %p\n", scause);
    printf("sepc=%p stval=%p\n", r_sepc(), r_stval());
    panic("kerneltrap");
  }
  // give up the CPU if this is a timer interrupt.
  if(which_dev == 2 && myproc() != 0 && myproc()->state == RUNNING)
    yield();

  w_sepc(sepc);
  w_sstatus(sstatus);
}
```

The idea is the same. `trapinit` and `trapinithart` do some initialization.

```c
void trapinit(void) {
  initlock(&tickslock, "time");
}

void trapinithart(void) {
  w_stvec((uint64)kernelvec);
}
```
