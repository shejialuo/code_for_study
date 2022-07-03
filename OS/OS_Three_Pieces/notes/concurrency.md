# Concurrency

## Lock

### Controlling Interrupts

One of the earliest solutions used to provide mutual exclusion was
to disable interrupts for critical sections; this solution was
invented for single-processor systems.

```c
void lock() {
  DisableInterrupts();
}
void unlock() {
  EnableInterrupts();
}
```

The main positive of this approach is its simplicity. The negatives are many.

+ This approach requires us to allow any calling thread to perform
a *privileged* operation, and thus *trust* that this facility is not
abused.
+ This approach does not work on multiprocessors.
+ Turning off interrupts for extended periods of time can lead to
interrupts
becoming lost, which can lead to serious system problems.
+ This approach can be inefficient.

For these reasons, turning off interrupts is only used in limited
contexts. For example, in some cases an operating system itself will use
interrupt masking to guarantee atomicity when accessing its own
data structures. This usage makes sense, as the trust issue disappears
inside the OS, which always trusts itself to perform privileged operations anyhow.

### A Failed Attempt: Just Using Loads/Stores

Let's first try to build a simple lock by using a single flag variable.

```c
typedef struct __lock_t {int flag;} lock_t;

void init(lock_t *mutex) {
  mutex->flag = 0;
}

void lock(lock_t *mutex) {
  while(mutex->flag == 1)
    ; // spin-wait (do nothing)
  mutex->flag = 1;
}

void unlock(lock_t *mutex) {
  mutex->flag = 0;
}
```

In this first attempt, the idea is quite simple: use a simple variable to
indicate whether some thread has possession of a lock. Unfortunately, the
code has two problems: one of correctness, and another of performance.
When the interruption happens at `while(mutex->flag == 1)`, another thread
could set `mutex->flag` to be 1, which is not correct.

The performance problem is the fact that the way a thread waits to acquire
a lock that is already held: it endlessly checks the value of a flag, a
technique known as *spin-waiting*.

### Building Working Spin Locks with Test-And-Set

The simplest bit of hardware support to understand is what is known as
a *test-and-set instruction*, also known as atomic exchange.

```c
int TestAndSet(int *old_ptr, int new) {
  int old = *old_ptr; // fetch old value at old_ptr
  *old_ptr = new; // store ’new’ into old_ptr
  return old; // return the old value
}
```

### Compare-And-Swap

Another primitive that some systems provide is known as the *compare-and-swap*
instruction, or *compare-and-exchange*.

```c
int CompareAndSwap(int *ptr, int expected, int new) {
  int actual = *ptr;
  if (actual == expected)
  *ptr = new;
  return actual;
}
```

### Just yield

Spinning isn't an efficient way. We should reschedule the process.
