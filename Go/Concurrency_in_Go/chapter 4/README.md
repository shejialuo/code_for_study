# 4. Concurrency Patterns in Go

## 4.1 Confinement

When working with concurrent code, there are a few different
options for safe operation. We've gone over two of them:

+ Synchronization primitives for sharing memory.
+ Synchronization via communicating.

However, there are a couple of other options that are implicitly safe
within multiple concurrent processes:

+ Immutable data
+ Data Protected by confinement

## 4.2 The for-select Loop

Something you'll see over and over again in Go programs is the
for-select loop. It's nothing more than something like this:

```go
for {
  select {

  }
}
```

There are a couple of different scenarios where you'll see this
pattern pop up.

*Sending iteration variables out on a channel*.

```go
for _, s = range []string{"a", "b", "c"} {
  select {
  case <-done:
    return
  case stringStream <-s
  }
}
```

*Looping infinitely waiting to be stopped*.

## 4.3 Preventing Goroutine Leaks

Goroutines are not garbage collected by the runtime, so regardless of
how small their memory footprint is, we don't want to leave them
lying about our process.

## 4.4 The or-channel

At times you may find yourself wanting to combine one or more `done` channels
into a single `done` channel that closes if any of its component channels close.
You can combine these channels together using the *or-channel* pattern.

This pattern creates a composite `done` channel through recursion
and goroutines.

## 4.5 Pipelines

A *pipeline* is just another tool you can use to form an abstraction in your
system. In particular, it is a very powerful tool to use when your
program needs to process streams, or batches of data.

Here is a function that can be considered a pipeline stage.

```go
multiply := func(values []int, multiplier int) []int {
  multipliedValues := make([]int, len(values))
  for i, v := range values {
    multipliedValues[i] = v * multiplier
  }
  return multipliedValues
}
```

Let's create another stage:

```go
add := func(vales []int, additive int) []int {
  addedValues := make([]int, len(values))
  for i, v := range values {
    addedValues[i] = v + additive
  }
  return addedValues
}
```

At this point, you might be wondering what makes these two functions
pipeline stages and not just functions. Let's try combining them:

```go
ints := []int{1, 2, 3, 4}
for _, v := range add(multiply(ints, 2), 1) {
  fmt.Println(v)
}
```

We constructed the functions to have the properties of a pipeline
stage, we're able to combine them to form a pipeline. That's interesting;
what *are* the properties of a pipeline stage?

+ A stage consumes and returns the same type.
+ A stage must be reified by the language so that it may be
passed around.

### Best Practices for Constructing Pipelines

Channels are uniquely suited to constructing pipelines in Go
because they fulfill all of our basic requirements. They can receive
and emit values, they can safely be used concurrently, they can
be ranged over, and they are reified by the language. Let' take
a moment and convert the previous example to utilize channels instead.

[Pipeline example](./pipelineExample.go)

### Some Handy Generators

A generator for a pipeline is any function that converts a
set of discrete values into a stream of values on a channel.

```go
repeat := func(
  done <-chan interface{}
  values ...interface{}
) <-chan interface{} {
  valueStream := make(chan interface{})
  go func() {
    defer close(valueStream)
    for {
      for _, v := range values {
        select {
        case <-done:
          return
        case valueStream <- v:
        }
      }
    }
  }()
  return valueStream
}
```

## 4.6 Fan-Out, Fan-In

So you've got a pipeline set up. Data is flowing through your system
beautifully, transforming a it make its way through the stages you've
chained together. It's like a beautiful stream. However, it is slow.

Sometimes, stages in your pipeline can be particularly computationally
expensive. When this happens, upstream stages in your pipeline can
become blocked whi waiting for your expensive stages to complete.

One of the interesting properties of pipelines is the ability they
give you to operate on the stream of data using a combination of
separate, often reorderable stages. You can parallelize the task.

In fact this pattern has a name: *fan-out, fan-in*. Fanning means *multiplexing*
or joining together multiple streams of data into a single stream.

```go
fanIn := func(
  done <-chan interface{},
  channels ...<-chan interface{},
) <-chan interface {
  var wg sync.WaitGroup
  multiplexedStream := make(chan interface{})

  multiplex := func(c <-chan interface{}) {
    defer wg.done()
    for i := range c {
      select {
      case <-done:
        return
      case multiplexedStream <- i:
      }
    }
  }

  wg.Add(len(channels))
  for _, c = range channels {
    go multiplex(c)
  }

  go func() {
    wg.Wait()
    close(multiplexedStream)
  }()

  return multiplexedStream
}
```

## 4.7 The context Package

As we've seen, in concurrent programs it's often necessary to preempt
operations because of timeouts, cancellation or failure of another
portion of the system. We've looked at the idiom of creating a `done`
channel, which flows through your program and cancels all blocking concurrent
operations.

It would be useful if we could communicate extra information alongside the
simple notification to cancel.

It turns ou that the need to wrap a `done` channel with this information
is very common in systems of any size, and so the Go authors decided
to create a standard pattern for doing so, which is called `context`.
