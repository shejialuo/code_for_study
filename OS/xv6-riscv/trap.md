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
