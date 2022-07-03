from auxiliary import Semaphore, Thread

n = 4 # the thread number

count = 0

mutex = Semaphore(1)

barrier = Semaphore(0)

def barrier_func():
    global count
    print('Hello Barrier Example')
    mutex.wait()
    count += 1
    mutex.signal()

    if count == n:
        """
        When the count equals to n, it should
        signal the barrier, thus could make one of
        the blocked thread become unblocked, thus
        executing `barrier.signal()` to unblock
        the others, thus all the threads become
        unblocked. But you may see the problem here,
        we could only use this barrier once, because
        for this situation, we just use `barrier.signal()`,
        don't use `barrier.wait()`
        which lets the value of semaphore become 1.
        """
        barrier.signal()
        print('All blocked thread continue')

    barrier.wait()
    barrier.signal()

    print('Done')

[Thread(barrier_func) for i in range(n)]
