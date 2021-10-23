from auxiliary import Semaphore

fork = [Semaphore(1) for i in range(5)]
footman = Semaphore(4)

def left(i):
    return i

def right(i):
    return (i + 1) % 5

def get_forks(i):
    footman.wait()
    fork[left(i)].wait()
    fork[right(i)].wait()

def put_forks(i):
    fork[left(i)].signal()
    fork[right(i)].signal()
    footman.signal()
