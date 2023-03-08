# 第三章 内核是如何与用户进程协作的

当网卡被送到协议栈后，需要通知到用户进程，让用户进程能收到并处理这些数据。进程和内核配合的方案有很多：

+ 同步阻塞。
+ 多路I/O复用。

## 3.1 socket的直接创建

对于开发者，调用`socket`函数既可以创建一个socket，如下图所示：

```c
int sk = socket(AF_INET, SOCK_STREAM, 0);
```

![socket内核结构](https://s2.loli.net/2023/03/08/tDW2Tq8SkPeK91h.png)

```c
SYSCALL_DEFINE3(socket, int, family, int, type, int, protocol) {
  .....
  retval = sock_create(family, type, protocol, &sock);
}
```

其中`sock_create`又调用了`__sock_create`：

```c
int __socket_create(struct net *net, int family, ...) {
  struct socket *sock;
  const struct net_proto_family *pf;

  // 分配socket对象
  socket = sock_alloc();

  // 获得每个协议簇的操作表
  pf = rcu_dereference(net_families[family]);

  // 调用每个协议栈的创建函数，将协议栈的处理方法赋值
  // 给socket
  err = pf->create(net, sock, protocol, kern);
}
```

`pf->create`会根据协议栈，调用相应的创建函数：

```c
static int inet_create(...) {
  // 将协议栈的处理方法赋值 给socket
  ...

  // 对sock对象进行初始化
  sock_init_data(sock ,sk);
}
```

`sock_init_data`会对`sock`中的`sk_data_ready`的函数指针进行初始化，设置为默认值：

```c
void sock_init_data(struct socket *sock, struct sock* sk) {
  sk->sk_data_ready = sock_def_readable;
  ...
}
```

当软中断上收到数据包时会调用`sk_data_ready`函数指针来唤醒sock上等待的进程。

## 3.2 内核和用户进程写协作之阻塞方式

### 3.2.1 等待接收消息

其调用的方式如下：`recvfrom`->`__sock_recvmsg` -> `__sock_recvmsg_nosec`然后调用sock中的`ops`的`recvmsg`方法，也就是协议层处理，即`inet_recvmsg`函数:

[recvfrom系统调用](https://s2.loli.net/2023/03/08/EuxN3pLVyoYUJCP.png)

```c
int inet_recvmsg(...) {
  err = sk->sk_prot->recvmsg(...);
}
```

然后继续调用协议层的方法，即`tcp_recvmsg`方法：

```c
int tcp_recvmsg(...) {
  int copied = 0;

  do {
    // 遍历接收队列接收数据
    skb_queue_walk(&sk->sk_receive_queue, skb) {
      ...
    }
  }

  if (copied >= target) {
    release_sock(sk);
    lock_sock(sk);
  } else {
    // 没有收到足够多的数据，阻塞当前进程。
    sk_wait_data(sk, &timeo);
  }

}
```

当内核收到数据后就绪事件产生，就可以查找`socket`等待队列上的等待项，进而可以找到回调函数和在等待该socket就绪事件的进程了。

### 3.2.2 软中断模块

软中断里收到数据包以后，发现是TCP包会执行`tcp_v4_recv`函数，然后放入对应的`socket`的接收队列中，最终调用`sk_data_ready`来唤醒用户进程。`sk_data_ready`会调用`wake_up_interruptible_sync_poll`来唤醒用户进程，取出等待项，执行其回调函数。

## 3.3 内核和用户进程协作之epoll

下面是一个简单的使用`epoll`的例子:

```c++
int main() {
  listen(lfd, ...);
  cfd1 = accept(...);
  cfd2 = accept(...);
  efd = epoll_create(...);

  epoll_ctl(efd, EPOLL_CTL_ADD, cfd1, ...);
  epoll_ctl(efd, EPOLL_CTL_ADD, cfd2, ...);
  epoll_wait(efd, ...);
}
```

+ `epoll_create`：创建一个epoll对象。
+ `epoll_ctl`：向epoll对象添加要管理的连接。
+ `epoll_wait`：等待其管理的连接上的I/O事件。

### 3.3.1 epoll内核对象的创建

在用户进程调用`epoll_create`时，内核会创建一个`struct eventpoll`的内核对象，并把它关联到当前进程已打开文件列表中。

```c
struct eventpoll {
  // Wait queue used by sys_epoll_wait()
  wait_queue_heat_t swq;

  // List of ready file descriptors
  struct list_head rdllist;

  // RB tree root used to store monitored fd structs
  struct rb_root rbr;

};
```

+ `wq`：等待队列链表。软中断数据就绪的时候会通过`wq`来找到阻塞在`epoll`对象上的用户进程。
+ `rbr`：一颗红黑树。为了支持海量连接的高效查找、插入和删除。管理用户进程下添加进来的所有socket连接。
+ `rdllist`：就绪的描述符的链表。当有连接就绪的时候，内核会把就绪的连接放到该链表中。

### 3.3.2 为epoll添加socket

为了简单起见，我们只考虑使用`EPOLL_CTL_ADD`添加socket,先忽略删除和更新。假设现在和客户端的多个连接的socket都已经创建好了，也创建好了epoll内核对象。在使用`epoll_ctl`注册每一个socket的时候，内核会做如下三件事情：

1. 分配一个红黑树节点对象`epitem`
2. 将等待事件添加到socket的等待队列中，其回调函数是`ep_poll_callback`。
3. 将`epitem`插入epoll对象的红黑树。

```c
SYSCALL_DEFINE4(epoll_ctl, int, epfd, int, op, int, fd,
               struct epoll_event __user *, event) {
  struct eventpoll *ep;
  struct file *file, *tfile;

  // 根据epfd找搭配eventpoll内核对象
  file = fget(epfd);
  ep = file->private_data;

  // 根据socket句柄号，找到其file内核对象
  tfile = fget(fd);

  switch(op) {
  case EPOLL_CTL_ADD:
    if (!epi) {
      epi.events |= POLLERR | POLLHUP;
      error = ep_insert(ep, &epds, tfile, fd);
    }
    ...
  }
}
```

然后其调用`ep_insert`代码实现注册，其首先需要创建并初始化`epitem`:

```c
struct epitem {
  // 红黑树节点
  struct rb_node rbn;

  // socket文件描述符信息
  // ffd->file 指向socket的内核文件
  // ffd->fd 指向socket的句柄
  struct epoll_filefd ffd;

  // 所归属的eventpoll对象
  struct eventpoll *ep;

  // 等待队列
  struct list_head pwqlist;
};
```

然后我们就可以看是如何进行插入的。

```c
static int ep_insert(struct eventpoll *ep,
                     struct epoll_event *event,
                     struct file *tfile, int fd) {
  // 1. 分配并初始化epitem
  struct epitem *epi;
  if (!(epi = kmem_cache_alloc(epi_cache, GFP_KERNEL)))
    return -ENOMEM;

  // 对分配的epi对象进行初始化
  INIT_LIST_HEAD(&epi->pwqlist);
  epi->ep = ep;
  ep_set_ffd(&epi->ffd, tfile, fd);

  // 2.设置socket等待队列
  // 定义并初始化ep_pqueue对象
  struct ep_pqueue epq;
  epq.epi = epi;
  init_poll_funcptr(&epq.pt, ep_ptable_queue_proc);

  // 调用ep_ptable_queue_proc注册回调函数
  // 实际注入的函数为ep_poll_callback
  revents = ep_item_poll(epi, &epq.pt);

  // 3. 将epi插入eventpoll对象的红黑树中
  ep_rbtree_insert(ep, epi);
}
```

最重要的操作位于`ep_item_poll`：

#### 设置socket等待队列

```c
static inline ep_item_poll(struct epitem *epi, poll_table *pt) {
  pt->key = epi->event.events;

  return epi->ffd.file->f_op->poll(epi->ffd.file, pt) & epi->event.events;
}
```

实际上是调用了socket下的`file->f_op->poll`:

```c
static unsigned int sock_poll(struct file *file, poll_table *wait) {
  ...
  return sock->ops->poll(file, sock, wait);
}
```

最终调用`tcp_poll`：

```c
unsigned int tcp_poll(struct file *file, struct socket *sock, poll_table *wait) {
  struct sock *sk = sock->sk;

  sock_poll_wait(file, sk_sleep(sk), wait);
}
```

`sk_sleep`获取了sock对象下的等待队列列表头`wait_queue_head_t`，然后调用`sock_poll_wait`进而调用`poll_wait`，然后调用注册了的`ep_ptable_queue_proc`函数。

该函数新建了一个等待队列项，并注册其回调函数为`ep_poll_callback`函数，然后将该等待项添加到socket的等待队列中。

### 3.3.3 epoll_wait的等待接收

其会判断`rdllist`链表中有没有数据，有数据就返回，没有就创建一个等待项，将其添加到`eventpoll`的等待队列中，然后将其自身阻塞。

其中最重要的就是`init_waitqueue_entry`：

```c
static int void init_waitqueue_entry(wait_queue_t *q, struct task_struct *p) {
  q->flags = 0;
  q->private = p;

  q->func = default_wake_function;
}
```

### 3.3.4 数据来了

在执行`epoll_ctl`的时候，内核为每一个socket都添加了一个等待队列项。在`epoll_wait`运行完时，又在`event poll`对象上添加了等待队列元素。

+ `socket->sock->sk_data_ready`设置的回调函数是`sock_def_readable`。
+ 在`socket`的等待队列中，其回调函数是`ep_poll_callback`。`private`指向空指针。
+ 在`eventpoll`的等待队列中，其回调函数是`default_wake_function`。`private`指向被阻塞的进程。

Linux提供了一个完美的抽象，对于`epoll`来说，其通过阻塞`epoll_wait`函数本身进行阻塞，而当有软中断到来时，其会唤醒`epoll_wait`本身，而`epoll`管理着某个进程所有的I/O操作。对于`recv`系统调用来说，`sock_def_readable`会唤醒进程本身，而对于`epoll`来说，其会执行`ep_poll_callback`，将`epitem`添加到`epoll`的就绪队列中，然后在等待队列中，调用`default_wake_function`，唤醒被阻塞的进程。

实际上，阻塞都是实际发生了的。而比起阻塞一个进程，我们更宁愿阻塞一个`epoll`操作，让这个`epoll`管理更多的进程，从而减少上下文的切换。
