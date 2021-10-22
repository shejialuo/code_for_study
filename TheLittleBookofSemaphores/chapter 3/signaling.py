from auxiliary import Semaphore, Thread

sem = Semaphore(0)

def print_for_thread_A():
    print('statement a')
    sem.signal()

def print_for_thread_B():
    sem.wait()
    print('statement b')

Thread(print_for_thread_B)
Thread(print_for_thread_A)
