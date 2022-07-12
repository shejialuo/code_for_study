# WorkQueue

## 令牌桶

client-go使用[令牌桶](https://en.wikipedia.org/wiki/Token_bucket)作为限流算法。

### Go标准库实现

#### Limiter数据结构定义

Go语言标准库提供了令牌桶算法的实现。首先在`rate.go`中定义了`Limiter`。

```go
type Limiter struct {
  mu     sync.Mutex
  limit  Limit
  burst  int
  tokens float64
  last time.Time
  lastEvent time.Time
}
```

其中`Limit`仅仅只是一个类型Wrapper：`type Limit float64`。字段的含义如下：

+ `mu`：互斥锁
+ `limit`：每秒下发令牌的个数
+ `burst`：桶的最大令牌数量
+ `tokens`：当前令牌数量
+ `last`：最后一次`tokens`字段更新时间
+ `lastEvent`：最近一次限流事件发生的时间

目前`limit`的定义为每秒下发令牌的个数，故`rate.go`定义了`Every`函数将事件之间的最小时间间隔转换为`Limit`。当`limit = Inf`时，`burst`可以被忽略，允许任何事件通过，因为下发令牌的个数是无限的。同时，`limit`也可以为0，代表不允许任何事件通过。

```go
func Every(interval time.Duration) Limit {
  if interval <= 0 {
    return Inf
  }
  return 1 / Limit(interval.Seconds())
}
```

`rate.go`定义了一些基本的getter和setter方法。

```go
func (lim *Limiter) Limit() Limit {
  lim.mu.Lock()
  defer lim.mu.Unlock()
  return lim.Limit
}

func (lim *Limiter) Burst() int {
  lim.mu.Lock()
  defer lim.mu.Unlock()
  return lim.burst
}

func NewLimiter(r Limit, b int) *Limiter {
  return &Limiter{
    limit: r,
    burst: b
  }
}
```

#### 辅助函数

为了更加好的抽象，`rate.go`定义了一系列的辅助函数。根据令牌桶的概念我们可以知道，随着时间的变化，令牌桶中的令牌数量会增加。故为了实现时间间隔和令牌桶的令牌数量相互的转化，`rate.go`定义了`tokensFromDuration`和`durationFromTokens`。

```go
// 得到一个时间段会产生多少个令牌
func (limit Limit) tokensFromDuration(d time.Duration) float64 {
  if limit <= 0 {
    return 0
  }
  return d.Seconds() * float64(limit)
}

// 目前令牌桶中的令牌代表了多少时间段
func (limit Limit) durationFromTokens(tokens float64) {
  if limit <= 0 {
    return InfDuration
  }
  seconds := tokens / float64(limit)
  return time.Duration(float64(time.Second) * seconds)
}
```

随着时间的变化，需要对令牌桶中的令牌也就是`token`进行更新，故定义了`advance`函数。

```go
func (lim *Limiter) advance(now time.Time) (newNow time.Time, newLast time.Time, newTokens float64) {
  last := lim.last
  if now.Before(last) {
    last = now
  }

  elapsed := now.Sub(last)
  delta := lim.limit.tokensFromDuration(elapsed)
  tokens := lim.tokens + delta
  if burst := float64(lim.burst); tokens > burst {
    tokens = burst
  }
  return now, last, tokens
}
```

#### 方法

Limiter还包含一些setter方法，介绍了辅助函数后，对于这些setter方法就比较容易理解。

```go
func (lim *Limiter) SetLimit(newLimit Limit) {
  lim.SetLimitAt(time.Now(), newLimit)
}

func (lim *Limiter) SetLimitAt(now time.Time, newLimit Limit) {
  lim.mu.Lock()
  defer lim.mu.Unlock()

  now, _, tokens := lim.advance(now)

  lim.last = now
  lim.tokens = tokens
  lim.limit = newLimit
}

func (lim *Limiter) SetBurst(newBurst int) {
  lim.SetBurstAt(time.Now(), newBurst)
}

// SetBurstAt sets a new burst size for the limiter.
func (lim *Limiter) SetBurstAt(now time.Time, newBurst int) {
  lim.mu.Lock()
  defer lim.mu.Unlock()

  now, _, tokens := lim.advance(now)

  lim.last = now
  lim.tokens = tokens
  lim.burst = newBurst
}
```

Limiter主要有三个方法：`Allow`, `Reserve`和`Wait`。这三个方法在被调用时，都会消耗掉一个令牌。这三个方法分别被`AllowN`，`ReserveN`以及`WaitN`抽象。

```go
func (lim *Limiter) Allow() bool {
  return lim.AllowN(time.Now(), 1)
}

func (lim *Limiter) Reserve() *Reservation {
  return lim.ReserveN(time.Now(), 1)
}

func (lim *Limiter) Wait(ctx context.Context) (err error) {
  return lim.WaitN(ctx, 1)
}
```

首先，`rate.go`定义了`Reservation`数据结构，包含了已经被限流器所允许的事件的信息。

```go
type Reservation struct {
  ok        bool
  lim       *Limiter
  tokens    int
  timeToAct time.Time
  limit Limit
}
```

字段的含义如下：

+ `ok`：表示事件能否发生
+ `lim`: 属于哪个Limiter
+ `tokens`：表示该事件需要消耗的令牌数量
+ `timeToAct`：执行的时间
+ `limit`：在`Reserve`操作的时候定义

我们首先看函数`AllowN`。

```go
func (lim *Limiter) AllowN(now time.Time, n int) bool {
  return lim.reserveN(now, n, 0).ok
}
```

再看函数`ReserveN`

```go
func (lim *Limiter) ReserveN(now time.Time, n int) *Reservation {
  r := lim.reserveN(now, n, InfDuration)
  return &r
}
```

可以看出`AllowN`和`ReserveN`都是通过`reserveN`进行抽象的。首先`reserveN`处理特殊情况，即`limit = Inf`（允许任何事件通过）和`limit = 0`（不允许任何事件通过），虽然不允许任何事件通过，但是本身令牌桶初始化時有`burst`个令牌数，故还是可以允许通过`burst`个令牌。

再处理完特殊情况后，首先通过`advance`计算出现在时刻的令牌桶中的令牌数量的个数，减去该事件所消耗的令牌个数。当令牌数小于0证明该事件需要等待，故通过`durationFromTokens`计算需要等待的时间。

其次，判断事件能否发生。事件能发生需要满足两个条件，一是事件发生消耗的令牌数量不能超过令牌桶最大的令牌数量，二是等待时间不能超过参数`maxFutureReserve`的值。

后面的操作就是更新字段。

```go
func (lim *Limiter) reserveN(now time.Time, n int, maxFutureReserve time.Duration) Reservation {
  lim.mu.Lock()
  defer lim.mu.Unlock()

  if lim.limit = Inf {
    return Reservation{
      ok:        true,
      lim:       lim,
      tokens:    n,
      timeToAct: now,
    }
  } else if lim.limit == 0 {
    var ok bool
    if lim.burst >= n {
      ok = true
      lim.burst -= n
    }
    return Reservation{
      ok:        true,
      lim:       lim,
      tokens:    lim.burst,
      timeToAct: now,
    }
  }

  now, last, tokens := lim.advance(now)

  tokens -= float64(n)

  var waitDuration time.Duration
  if tokens < 0 {
    waitDuration = lim.limit.durationFromTokens(-tokens)
  }

  ok := n <= lim.burst && waitDuration <= maxFutureReserve

  r := Reservation{
    ok:    ok,
    lim:   lim,
    limit: lim.limit,
  }
  if ok {
    r.tokens = n
    r.timeToAct = now.Add(waitDuration)
  }

  if ok {
    lim.last = now
    lim.tokens = tokens
    lim.lastEvent = r.timeToAct
  } else {
    lim.last = last
  }

  return r
}
```

可以看出`reserveN`作为一个核心的函数，无非就是查询令牌桶中的令牌数量足不足以支持一个消耗`n`个令牌的任务，为了维持这个任务的状态必须定义一个数据结构来维持。

在讲`WaitN`函数之前，我们先看看`DelayFrom`函数，这个函数很简单，对于已经ok的任务，得到其延迟发生的时间。

```go
func (r *Reservation) DelayFrom(now time.Time) time.Duration {
  if !r.ok {
    return InfDuration
  }
  delay := r.timeToAct.Sub(now)
  if delay < 0 {
    return 0
  }
  return delay
}
```

现在我们可以去看`WaitN`函数，很显然对于一个任务来说，其可以通过调用`WaitN`来实现限流。

+ 当所需的令牌数目大于令牌桶所能包含的最大令牌，直接返回error。
+ 如果在调用时，任务已经结束了，直接返回error。
+ 计算`waitLimit`其值为任务结束的时间和现在的时间的差值，然后使用`reserveN`得到任务的状态。
+ 然后使用`DelayFrom`计算需要延迟的时间，如果有必要延迟的话，通过一个定时器来延时。如果定时器完成了，就继续。如果定时器结束之前，`Context`被取消了，返回错误。

```go
func (lim *Limiter) WaitN(ctx context.Context, n int) (err error) {
  lim.mu.Lock()
  burst := lim.burst
  limit := lim.limit
  lim.mu.Unlock()

  if n > burst && limit != Inf {
    return fmt.Errorf("rate: Wait(n=%d) exceeds limiter's burst %d", n, burst)
  }
  select {
  case <-ctx.Done():
    return ctx.Err()
  default:
  }
  now := time.Now()
  waitLimit := InfDuration
  if deadline, ok := ctx.Deadline(); ok {
    waitLimit = deadline.Sub(now)
  }
  r := lim.reserveN(now, n, waitLimit)
  if !r.ok {
    return fmt.Errorf("rate: Wait(n=%d) would exceed context deadline", n)
  }
  delay := r.DelayFrom(now)
  if delay == 0 {
    return nil
  }
  t := time.NewTimer(delay)
  defer t.Stop()
  select {
  case <-t.C:
    return nil
  case <-ctx.Done():
    r.Cancel()
    return ctx.Err()
  }
}
```

注意`r.Cancel()`的使用，既然我们已经给了令牌给一个任务而这个任务并没有实际的执行，我们应该还给令牌桶相应的数目。由于此时已经介绍了大部分的函数，此处忽略其细节。

#### 小结

令牌桶的实现与时间有很大的关系，看似需要每隔1s就需要更新令牌桶中的令牌数目，实则上是完全没有必要的。因为可以从未来借。当每一次调用主要方法时，都会通过现在的时间减去上一次令牌桶数目更新的时间来更新令牌桶中的令牌数目，令牌桶中的令牌数目是负的也根本无所谓，很棒的设计。

### Client-go封装

Client-go在`util/workqueue/default_rate_limiters.go`中定义了`BucketRateLimiter`用于封装标准库中的`Limiter`。

```go
type BucketRateLimiter struct {
  *rate.Limiter
}
```
