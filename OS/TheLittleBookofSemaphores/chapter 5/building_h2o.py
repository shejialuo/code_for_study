from auxiliary import Semaphore

hydrogen_num = 0
oxygen_num = 0
mutex = Semaphore(1)
hydrogen_queue = Semaphore(0)
oxygen_queue = Semaphore(0)
barrier = Semaphore(3)

def hydrogen_thread():
    global oxygen_num, hydrogen_num
    mutex.wait()
    hydrogen_num += 1
    if hydrogen_num >= 2 and oxygen_num >= 1:
        hydrogen_queue.signal(2)
        hydrogen_num -= 2
        oxygen_queue.signal()
        oxygen_num -= 1
    else:
        mutex.signal()
    hydrogen_queue.wait()
    barrier.wait()
    mutex.signal()

def oxygen_thread():
    global oxygen_num, hydrogen_num
    mutex.wait()
    oxygen_num += 1
    if hydrogen_num >= 2:
        hydrogen_queue.signal(2)
        hydrogen_num -= 2
        oxygen_queue.signal()
        oxygen_num -= 1
    else:
        mutex.signal()

    oxygen_queue.wait()
    barrier.wait()
