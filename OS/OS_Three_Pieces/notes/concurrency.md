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
