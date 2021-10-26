# In this situation, the writer can be starved

from auxiliary import Semaphore, Thread

readers = 0
mutex = Semaphore(1)
roomEmpty = Semaphore(1)

data = [i for i in range(10)]

def writer_thread_func(num):
    roomEmpty.wait()
    data.append(num)
    roomEmpty.signal()

def reader_thread_func(index):
    global readers

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