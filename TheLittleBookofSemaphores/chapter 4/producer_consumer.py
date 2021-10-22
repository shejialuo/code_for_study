from auxiliary import Semaphore, Thread

buffer = 5
amount = 0

items = Semaphore(0)
spaces = Semaphore(buffer)
mutex = Semaphore(1)

line = [0 for i in range(5)]

def producer(num):
    global amount
    spaces.wait()
    mutex.wait()
    line[amount] = num
    amount += 1
    mutex.signal()
    items.signal()

def consumer():
    global amount
    items.wait()
    mutex.wait()
    print(line[amount - 1])
    amount -= 1
    mutex.signal()
    spaces.signal()

Thread(consumer)
Thread(producer, 3)
