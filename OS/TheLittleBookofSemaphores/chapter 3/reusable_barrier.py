from auxiliary import Semaphore, Thread

mutex = Semaphore(1)
turnstile = Semaphore(0)
turnstile2 = Semaphore(1)

count = 0
n = 4

#! Two-phase locker
def reusbale_barrier_func():
    print("Hello Reusable Barrier")
    global count

    mutex.wait()
    count += 1
    if count == n:
        turnstile2.wait()    # lock the second
        turnstile.signal()   # unlock the first
    mutex.signal()

    turnstile.wait()         # first turnstile
    turnstile.signal()

    # At this point, the value of turnstile is 1.

    print("Critical Section")

    mutex.wait()
    count -= 1
    if count == 0:
        turnstile.wait()     # To make turnstile to 0 for next iteration
        turnstile2.signal()  # unlock the second
    mutex.signal()

    turnstile2.wait()        # second turnstile
    turnstile2.signal()
