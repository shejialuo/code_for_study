# Writer has more priority than Reader

# In this situation, we see all writers as a queue
# and also see all readers as a queue.

from auxiliary import Semaphore, Thread

class LightSwitch():

    def __init__(self):
        self.counter = 0
        self.mutex = Semaphore(1)
    
    def lock(self, semaphore):
        self.mutex.wait()
        self.counter += 1
        if self.counter == 1:
            semaphore.wait()
        self.mutex.signal()

    def unlock(self,semaphore):
        self.mutex.wait()
        self.counter -= 1
        if self.counter == 0:
            semaphore.signal()
        self.mutex.signal()

noReaders = Semaphore (1)
noWriters = Semaphore (1)

readSwitch = LightSwitch()
writeSwitch = LightSwitch()

data = [i for i in range(10)]

def writer_thread_func(num):

    # When the first writer thread run,
    # it blocks any readers and maintain 
    # a writer thread queue.
    writeSwitch.lock(noReaders)

    # noWriters as a mutex, also as a conditional variable
    noWriters.wait()

    data.append(num)
    noWriters.signal()

    writeSwitch.unlock(noReaders)

def reader_thread_func(index):
    global readers

    # noReads as a mutex, but also as a conditional variable
    # for writer, wonderful!
    noReaders.wait()

    # When the first reader thread run, block any writers.
    # And maintain a reader thread queue.
    readSwitch.lock(noWriters)

    noReaders.signal()

    try:
        print(data[index])
    except IndexError:
        print('There is no corresponding index')

    readSwitch.unlock(noWriters)

Thread(reader_thread_func, 2)
Thread(reader_thread_func,3)
Thread(writer_thread_func, 11)
Thread(reader_thread_func, 11)