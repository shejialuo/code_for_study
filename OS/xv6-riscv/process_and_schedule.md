# Process and Schedule

## Process Basic

`kvmmake` in `vm.c` calls `proc_mapstacks` to allocate a page
for each process's kernel stack.

```c
void proc_mapstacks(pagetable_t kpgtbl) {
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}
```

You may wonder why we need to allocate page for each process, and use
`KSTACK` to get a higher virtual address. When scheduling, the
kernel must maintain each process' state. That' why in PCB, xv6
defines `kstack`.

Now, the xv6 uses `procinit` to initialize the process table.

```c
void procinit(void) {
  struct proc *p;

  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  for(p = p->proc, p < &proc[NPROC]; p++) {
    initlock(&p->lock, "proc");
    p->kstack = KSTACK((int) (p - proc));
  }
}
```

Well, other than the initialization of the locks, the `procinit`
initializes the `p->kstack`.

How do we know what is the current running process? xv6 defines `myproc`.

```c
struct proc* myproc(void) {
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}
```

xv6 uses `mycpu` to get the current CPU.

```c
struct cpu* mycpu(void) {
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}
```

Well for `cpuid`, it is just a wrapper assembly code for get the
current cpu id. I omit detail here. So we can get the idea: xv6 uses `struct cpu`
to hold the information on the CPU.

```c
struct cpu {
  struct proc *proc;
  struct context context;
  int noff;
  int intena;
}
```

We can see that the each cpu holds the `struct context`.

The strategy to produce the PID is simple easy in xv6. It uses add-one.

```c
int allocpid() {
  int pid;

  acquire(&pid_lock);
  pid = nextpid;
  nextpid = nextpid + 1;
  release(&pid_lock);

  return pid;
}
```

And we also want to find a way to allocate a new process.

```c
static struct proc* allocproc(void) {
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  return p;
}
```

When allocating a new process, first, we need to find whether there is a
process that we could use. If we find, we first use `allocpid` to get the
new pid and then set the process state to be `USED`. In order to
initialize the trapframe, we use `kalloc` to get a page-size memory.
If failed, we should call `freeproc`.

```c
static void
freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
}
```

Well, when free a proc, we should free the `trapframe` and its pagetable.

In the process of allocating a process, we should request a pagetable
for the process, this is what `proc_pagetable` does.

```c
pagetable_t proc_pagetable(struct proc *p) {
  pagetable_t pagetable;

  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }
}
```

Well, it is easy to understand. The most important thing is that
we map the `TRAMPOLINE` in the user process' pagetable for trapping.

For freeing the pagetable, we should undo the map and free the memory.

```c
void
proc_freepagetable(pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME, 1, 0);
  uvmfree(pagetable, sz);
}
```

In the end of the allocation a process, xv6 sets the `context.ra` to
`forkret` and set the stack pointer to the top of the `kstack`.

```c
void forkret(void) {
  static int first = 1;

  release(&myproc()->lock);
  if(first) {
    first = 0;
    fsinit(ROOTDEV);
  }

  usertrapret();
}
```

Well, here `forkret` uses `usertrapret()` to transfer from kernel to user.

Now it's time to start our first user process.

```c
void userinit(void) {
  struct proc *p;
  p = allocproc();
  initproc = p;

  uvminit(p->pagetable, initcode, sizeof(initcode));
  p->sz = PGSIZE;
  p->trapframe->epc = 0;
  p->trapframe->sp = PGSIZE;

  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");

  p->state = RUNNABLE;

  release(&p->lock);
}
```

## Schedule

xv6 defines `scheduler` function to yield the CPU and
switch to the other RUNNABLE process.

```c
void scheduler(void) {
  struct proc *p;
  struct cpu *c = mycpu();

  c->proc = 0;
  for(;;) {
    intr_on();
    for(p = proc; p < &proc[NPROC]; p++) {
      acquire(&p->lock);
      if(p->state = RUNNABLE) {
        p->state = RUNNING;
        c->proc = p;
        switch(&c->context, &p->context);
        c->proc = 0;
      }
      release(&p->lock);
    }
  }
}
```

`switch` function is defined in `switch.S` and it is super easy because
of the simplicity of the RISC-V. However, you may be confused about the
`c->proc = 0`. When we have finished the `switch` function, the
cpu is going to execute the process' code. So we must let the control
come back to the scheduler when the time interrupt happens.
Remember in `trap.c`, when the time interrupt happens, it calls `yield` to give
up the CPU.

```c
void yield(void) {
  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  sched();
  release(&p->lock);
}
```

xv6 uses `sched` to switch to scheduler.

```c
void sched(void) {
  int intena;
  struct proc *p = myproc();
  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched running");
  if(intr_get())
    panic("sched interruptible");
  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}
```

However, here we need to talk about more detail. Actually, the `sched` and
`scheduler` are cooperating together to finish the work. When using `scheduler`,
the `switch` stores return address into `struct context`, and when using
`sched`, the `switch` restores the cpu return address, thus the flow goes back
to the `scheduler`. The same is as the process, when using `sched`, the `switch`
stores return address into `struct context`, and when using `scheduler`, the
`switch` restores the process return address, thus the flow goes back to the
`sched`.

And all these opeartions happened at the time interruption which is transparent
to the scheduling. Acutally, scheduling is just about to storing the context. But
I think this coordinator is GOOD way. I have learned much about it.

### Summary

Well, the schduler functionality is achieved by the `struct context`.
Our kernel is also a process. So each cpu also incoperates the
`struct context`. When the kernel schedule a new process, kernel
finds the RUNNABLE status and save the current context to
the CPU's `struct context`. And the make the current context to
be the process's `context`. Now the exeuction flow is transfer to
the process. When time interruption happened, the kernel first
makes sure that the current stage is stored in the `struct trapframe`,
and uses `yield` to give up the CPU and returns to the
`scheduler` to
run another process.

## Process Operation

Like Linux, xv6 defeins the `fork` to create a new process and
copy the parent.

```c
int fork(void) {
  int i, pid;
  struct proc *np;
  strucr proc *p = myproc();

  if((np = allocproc()) == 0) {
    return -1;
  }
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  *(np->trapframe) = *(p->trapframe);

  np->trapframe->a0 = 0;
  for(i = 0; i < NOFILE; i++)
    if(p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);
  safestrcpy(np->name, p->name, sizeof(p->name));
  pid = np->pid;
  release(&np->lock);

  acquire(&wait_lock);
  np->parent = p;
  release(&wait_lock);

  acquire(&np->lock);
  np->state = RUNNABLE;
  release(&np->lock);

  return pid;
}
```
The interesting part is that the `fork` uses `np->trapframe->a0 = 0` to
return 0 in the child. It is simple enough.

When the children are abandoned, we should make its parent to be `init`.

```c
void reparent(struct proc* p) {
  struct proc* pp;
  for(pp = proc; pp < &proc[NPROC]; pp++) {
    if(pp->parent = p) {
      pp->parent = initproc;
      wakeup(initproc);
    }
  }
}
```
