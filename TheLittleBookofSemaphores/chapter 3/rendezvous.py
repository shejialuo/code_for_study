from auxiliary import Semaphore, Thread

a_arrived = Semaphore(0)
b_arrived = Semaphore(0)

def print_for_thread_A():
    print('statement a1')
    a_arrived.signal()
    b_arrived.wait()
    print('statement a2')

def print_for_thread_B():
    print('statement b1')
    b_arrived.signal()
    a_arrived.wait()
    print('statement b2')

Thread(print_for_thread_A)
Thread(print_for_thread_B)
