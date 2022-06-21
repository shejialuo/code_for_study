# Process Data Structure

The kernel should have a way to manage the process. There are
some many parts: initialization, execute, and schedule.
All these parts need some data structure. This note is aimed to
understanding the data structure.

## Context

Due to the simple architecture of the RISC-V. The register we need to
save is simple enough. In `proc.h`, xv6 defines `context` as follows.

```c
struct context {
  uint64 ra;
  uint64 sp;

  uint64 s0;
  uint64 s1;
  uint64 s2;
  uint64 s3;
  uint64 s4;
  uint64 s5;
  uint64 s6;
  uint64 s7;
  uint64 s8;
  uint64 s9;
  uint64 s10;
  uint64 s11;
};
```

It's so elegant compared with x86.

## PCB

In order to manage the process, kernel should maintain the process status.
In `proc.h`, xv6s define `proc`.

```c
struct proc {
  struct spinlock lock;

  // p->lock must be held when using these:
  enum procstate state;        // Process state
  void *chan;                  // If non-zero, sleeping on chan
  int killed;                  // If non-zero, have been killed
  int xstate;                  // Exit status to be returned to parent's wait
  int pid;                     // Process ID

  // wait_lock must be held when using this:
  struct proc *parent;         // Parent process

  // these are private to the process, so p->lock need not be held.
  uint64 kstack;               // Virtual address of kernel stack
  uint64 sz;                   // Size of process memory (bytes)
  pagetable_t pagetable;       // User page table
  struct trapframe *trapframe; // data page for trampoline.S
  struct context context;      // swtch() here to run process
  struct file *ofile[NOFILE];  // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)
};
```

From the code above, we can see that the definition of PCB is
easy to understand.

And for process state:

```c
enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };
```

Pay attention to the `struct trapframe`, it is important,
this data structure is the bridge between the user and the kernel.
