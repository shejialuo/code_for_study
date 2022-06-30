# Lock

xv6 has two types of locks: spin-lock and sleep-lock.

## Spin-Lock

In `spinlock.c`, xv6 defines `initlock` to initialize the lock.

```c
void initlock(struct spinlock *lk, char *name) {
  lk->name = name;
  lk->locked = 0;
  lk->cpu = 0;
}
```

For the lock, there are two common operations: acquire the lock
and release the lock.

```c
void acquire(struct spinlock *lk) {
  push_off();
  if(holding(lk))
    panic("acquire")
  while(__sync_lock_test_and_set(&lk->locked, 1) != 0)
    ;
  _sync_synchronize();
  lk->cpu = mycpu();
}
```

Well, xv6 don't allow acquire the lock which is already hold
by others.

```c
int holding(struct spinlock* lk) {
  int r;
  r = (lk->locked && lk->cpu == mycpu());
  return r;
}
```

Well, the lock is easy to understand, which uses compare and return lock
semantics. And the `release` is just like the `acquire`.

And for convenience, xv6 wraps `push_off` and `pop_off` to better
use `intr_off` and `intr_on`. But the idea is like stack. It
takes two `pop_off` to undo two `push_off`.

```c
void push_off(void) {
  int old = intr_get();

  intr_off();
  if(mycpu()->noff == 0)
    mycpu()->intena = old;
  mycpu()->noff += 1;
}

void
pop_off(void) {
  struct cpu *c = mycpu();
  if(intr_get())
    panic("pop_off - interruptible");
  if(c->noff < 1)
    panic("pop_off");
  c->noff -= 1;
  if(c->noff == 0 && c->intena)
    intr_on();
}
```

## Sleep-Lock

Sleep-lock incorporates the spin-lock. However, when understanding it,
we should first understand process and scheduling. So I omit detail here.
