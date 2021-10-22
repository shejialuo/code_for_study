from auxiliary import Semaphore, Thread

count = 0

mutex = Semaphore(1)

def count_thread_A():
    global count
    mutex.wait()
    count += 1
    mutex.signal()

def count_thread_B():
    global count
    mutex.wait()
    count += 1
    mutex.signal()

Thread(count_thread_A)
Thread(count_thread_B)

print(count)
