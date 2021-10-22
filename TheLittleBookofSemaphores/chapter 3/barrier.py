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
        barrier.signal()
        print('All blocked thread continue')

    barrier.wait()
    barrier.signal()

    print('Done')

[Thread(barrier_func) for i in range(n)]
