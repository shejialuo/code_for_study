from auxiliary import Semaphore, Thread

servings = 0
mutex = Semaphore(1)
empty = Semaphore(0)
full = Semaphore(0)

def savage():
    while True:
        global servings
        mutex.wait()
        if servings == 0:
            empty.signal()
            full.wait()
            servings = 10
        servings -= 1
        mutex.signal()
        print('Get food from pot')

def cook():
    while True:
        global servings
        empty.wait()
        print('Put food in pot')
        full.signal()

Thread(cook)
Thread(savage)
