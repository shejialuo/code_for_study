# System Call

## Kernel

Well, we have talked about the trap functionality. Now, it's time to
understand how system call works. The `usertrap` in `trap.c` calls `syscall` to
get the job done.

```c
void syscall(void) {
  int num;
  struct proc *p = myproc();

  num = p->trapframe->a7;
  if(num > 0 && num < NELEM(syscalls) && syscalls[num]) {
    p->trapframe->a0 = syscalls[num]();
  } else {
    printf("%d %s: unknown sys call %d\n",
            p->pid, p->name, num);
    p->trapframe->a0 = -1;
  }
}
```

Well, it may seem strange why we need to find the `a7` number, this
is what the ABI defines. So we can get the number of the function call. So
xv6 maintains an array of pointers to function. This may seem impossible,
because it is easy to think the system call's type should be different
between each other. Well, remember all the information is stored
in the `p->trapframe`. So we could have the same prototype which
is `void (*)(void)`.

```c
static uint64 (*syscalls[])(void) = {
[SYS_fork]    sys_fork,
[SYS_exit]    sys_exit,
[SYS_wait]    sys_wait,
[SYS_pipe]    sys_pipe,
[SYS_read]    sys_read,
[SYS_kill]    sys_kill,
[SYS_exec]    sys_exec,
[SYS_fstat]   sys_fstat,
[SYS_chdir]   sys_chdir,
[SYS_dup]     sys_dup,
[SYS_getpid]  sys_getpid,
[SYS_sbrk]    sys_sbrk,
[SYS_sleep]   sys_sleep,
[SYS_uptime]  sys_uptime,
[SYS_open]    sys_open,
[SYS_write]   sys_write,
[SYS_mknod]   sys_mknod,
[SYS_unlink]  sys_unlink,
[SYS_link]    sys_link,
[SYS_mkdir]   sys_mkdir,
[SYS_close]   sys_close,
};
```

Also, in order to fetch the value or the address, xv6 defines
many auxiliary functions to get the system call argument.

```c
int argint(int n, int *ip) {
  *ip = argraw(n);
  return 0;
}
```

```c
int argaddr(int n, uint64* ip) {
  *ip = argraw(n);
  return 0;
}
```

As you can see, `argraw` is used to get the `p->trapframe->n`:

```c
static uint64 argraw(int n) {
  struct proc *p = myproc();
  switch (n) {
  case 0:
    return p->trapframe->a0;
  case 1:
    return p->trapframe->a1;
  case 2:
    return p->trapframe->a2;
  case 3:
    return p->trapframe->a3;
  case 4:
    return p->trapframe->a4;
  case 5:
    return p->trapframe->a5;
  }
  panic("argraw");
  return -1;
}
```

Well, things become more complicated when fetching the arguments
as string. xv6 defines `argstr` to handle it.

```c
int argstr(int n, char *buf, int max) {
  uint64 addr;
  if(argaddr(n, &addr) < 0)
    return -1;
  return fetcstr(addr, buf, max);
}
```

Well, actually we just get the start address of the string and
then use `fetchstr` to store the string to `buf`.

```c
int fetchstr(uint64 addr, char *buf, int max) {
  struct proc *p = myproc();
  int err = copyinstr(p->pagetable, buf, addr, max);
  if(err < 0)
    return err;
  return strlen(buf);
}
```

`copyinstr` is easy. it needs first translate the virtual address `addr`
to physical address, and well, just copy.

And, the `fetchaddr` fetch the value at the specified address, the idea
is the same as `fetchstr`. I omit detail here.

## User

It is easy for user to do the system call. xv6 uses perl to
automatically generate assembly code.

```perl
sub entry {
    my $name = shift;
    print ".global $name\n";
    print "${name}:\n";
    print " li a7, SYS_${name}\n";
    print " ecall\n";
    print " ret\n";
}
```
