# 3 Go's Concurrency Building Blocks

## 3.1 Goroutines

Goroutines are one of the most basic units of organization in a Go
program, so it's important we understand what they are and how they work.
In fact, every Go program has at least one goroutine: the *main routine*,
which is automatically created and started when the process begins.

Goroutines are unique to Go. They're not OS threads, and they're not
exactly green threads, they're a higher level of abstraction known
as *coroutines*. Coroutines are simply concurrent subroutines
that are *nonpreemptive*. Instead, coroutines have multiple
points throughout which allow for suspension or reentry.

What makes goroutines unique to Go are their deep integration with
Go's runtime. Goroutines don't define their own suspension or
reentry points; Go's runtime observes the runtime behavior of
goroutines and automatically suspends them when they block and
then resumes them when they become unblocked.

Go follows a model of concurrency called the *fork-join* model.

## 3.2 The sync Package

The `sync` package contains the concurrency primitives that are
most useful for low-level memory access synchronization.

### 3.2.1 WaitGroup

`WaitGroup` is a great way to wait for a set of concurrent operations
to complete when you either don't care about the result of
the concurrent operation.

### 3.2.2 Mutex and RWMutex

If you're already familiar with languages that handle concurrency through
memory access synchronization, then you'll probably immediately
recognize `Mutex`

### 3.2.3 Cond

Just like the `pthread_cond`.

### 3.2.4 Once

`sync.Once` is a type that utilizes some `sync` primitives internally to
ensure that only one call to `Do` ever calls the function passed in
even on different goroutines.

## 3.3 Channels

Channels are one of the synchronization primitives in Go derived
from CSP. Like a river, a channel serves as a conduit for a stream
of information; values may be passed along the channel, and
then read out downstream. When using channels, you'll pass a value into
a `chan` variable, and then somewhere else in your program read it
off the channel.

Creating a channel is very simple.

```go
var dataStream chan interface{}
dataStream = make(chan interface{})
```

This example defines a channel, `dataStream`, upon which any value can
be written or read (because we used the empty interface). Channels
can be declared to only support a unidirectional flow of data.

To declare a unidirectional channel, you'll simply include the `<-`
operator. To both declare and instantiate a channel that can only
read, place the `<-` operator on the left-hand side.

```go
var dataStream <-chan interface{}
dataStream := make(<-chan interface{})
```

And to declare and create a channel that can only send, you place
the `<-` operator on the right-hand side.

```go
var dataStream chan<- interface{}
dataStream := make(chan<- interface{})
```

You don't often see unidirectional channels instantiated, but
you'll often see them used as function parameters and return types,
which is very useful. This is because Go will implicitly convert
bidirectional channels to unidirectional channels when needed.

```go
var receiveChan <-chan interface{}
var sendChan chan<- interface{}

dataStream := make(chan interface{})

receiveChan = dataStream
sendChan = dataStream
```

Keep in mind that channels are typed.

```go
intStream := make(chan int)
```

To use channels, we'll once again make use of the `<-` operator.
Sending is done by placing the `<-` operator to the right of a channel,
and receiving is done by placing the `<-` operator to the left of the channel.

We could perform reads on closed channel indefinitely despite the channel
remaining closed. This is to allow support for multiple downstream
reads from a single upstream writer on the channel.

We can also create *buffered channels*, which are channels
that are given a *capacity* when they're instantiated.

```go
var dataStream chan interface{}
dataStream = make(chan interface{}, 4)
```

## 3.4 The select Statement

The `select` statement is the glue that binds channels together;
it's how w're able to compose channels together in a program to form
larger abstractions.

```go
var c1, c2 <-chan interface{}
var c3 chan<- interface{}
select {
case <- c1:
  // do something
case <- c2:
  // do something
case c3 <- struct{}{}:
  // do something
}
```

All channel reads and writes are considered simultaneously to see
if any of them are ready.
