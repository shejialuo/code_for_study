# In reader_writer_1, the problem is that
# when there are continuous readers, writer
# could be blocked forever.
# So we can add another semaphore to deal
# with this situation.

# However, there is no priority for writer

from auxiliary import Semaphore, Thread

readers = 0
mutex = Semaphore(1)
roomEmpty = Semaphore(1)
turnstile = Semaphore(1)

data = [i for i in range(10)]

def writer_thread_func(num):
    turnstile.wait()
    roomEmpty.wait()
    data.append(num)
    roomEmpty.signal()
    turnstile.signal()

def reader_thread_func(index):
    global readers

    turnstile.wait()
    turnstile.signal()

    mutex.wait()
    readers += 1
    # The first reader should wait for the roomEmpty
    if (readers == 1):
        roomEmpty.wait()
    mutex.signal()
    try:
        print(data[index])
    except IndexError:
        print('There is no corresponding index')
    
    mutex.wait()
    readers -= 1
    # When there is no reader should signal
    if (readers == 0):
        roomEmpty.signal()
    mutex.signal()

Thread(reader_thread_func, 2)
Thread(reader_thread_func,3)
Thread(writer_thread_func, 11)
Thread(reader_thread_func, 11)