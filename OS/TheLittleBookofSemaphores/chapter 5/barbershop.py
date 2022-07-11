from auxiliary import Semaphore, Thread

n = 4
customers = 0
mutex = Semaphore(1)
customer = Semaphore(0)
barber = Semaphore(0)

customerDone = Semaphore(0)
barberDone = Semaphore(0)

def customer_thread():
    global customers
    mutex.wait()

    if customers == n:
        mutex.signal()
        # When the barbershop is full of customers,
        # the customer should just return
        return
    customers += 1
    mutex.signal()

    # We should signal the barber to cut the hair cut
    customer.signal()
    # We should wait if there is customer cutting the hair
    barber.wait()

    print("I am enjoying the service")

    # two rendezvouses
    customerDone.signal()
    barberDone.wait()

    mutex.wait()
    customers -= 1
    mutex.signal()

def barber_thread():
    while True:
        # When there is no customer, the barber
        # should go to sleep
        customer.wait()
        # We should signal the customer
        barber.signal()

        print("I am cutting the hair for the customer")

        # two rendezvouses
        customerDone.wait()
        barberDone.signal()

Thread(barber_thread)

[Thread(customer_thread) for _ in range(50)]
