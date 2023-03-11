# 第四章

## 4.1 网络包发送过程总览

发送的过程和接收的过程是类似的，用户数据被拷贝到内核态，然后经过协议栈处理后进入`RingBuffer`。随后网卡驱动真正将数据送了出去。当发送完成时，通过硬件中断处理CPU，触发软中断`NET_RX_SOFTIRQ`释放内存。

## 4.2 网卡启动过程

类似接收，此处忽略。

## 4.3 数据从用户进程到网卡的详细过程

`send`系统调用主要实现了两个功能：

### 4.3.1 send系统调用实现

1. 找出socket对象。
2. 构建`struct msghdr`对象，传入用户的数据等信息。

```c
SYSCALL_DEFINE6(...) {
  // 根据fd找到socket
  sock = sockfd_lookup_light(fd, &err, &fput_needed);

  // 构建struct msghdr
  struct msghdr mgs;
  struct iovec iov;

  iov.iov_base = buff;
  iov.iov_len = len;
  msg.msg_iovlen = 1;

  msg.msg_iov = &iov;
  msg.msg_flags = flags;

  // 发送数据
  sock_sendmsg(sock, &msg, len);
}
```

然后进入协议栈。

### 4.3.2 传输层处理

在进入协议栈后，内核会找到socket对象上的具体协议发送函数。对于TCP协议来说，就是`tcp_sendmsg`。在这个函数中，内核会申请一个内核态的skb内存，将用户待发送的数据拷贝进去。

```c
int inet_sendmsg(...) {
  return sk->sk_port->sendmsg(iocb, sk, msg, size);
}
```

```c
int tcp_sendmsg(...) {
  while (...) {
    while (...) {
      // 获取发送队列
      skb = tco_write_queue_tail(sk);

      // 申请skb并拷贝
    }
  }
}
```

然后`tcp_sendmsg`会处理用户传入的`struct msghdr`，将用户数据里的内存拷贝到内核态。并根据滑动窗口判断是否应该发送数据。

```c
int tcp_sendmsg(...) {
  while (...) {
    while (...) {
      if (forced_push(tp)) {
        tcp_mark_push(tp, skb);
        __tcp_push_pending_frames(sk, mss_now, TCP_NAGLE_PUSH);
      } else if (skb == tcp_send_head(sk)) {
        tcp_push_one(sk,mss_now);
      }
      continue;
    }
  }
}
```

当满足真正发送条件后，最终到会执行到`tcp_write_xmit`:

```c
static bool tcp_write_xmit(struct sock *sk, unsigned int mss_now, int nonangle,
                          int push_one, gfp_t gfp) {
  while ((skb = tcp_send_head(sk))) {
    // 滑动窗口处理
    ...
    // 开启发送
    tcp_transmit_skb(sk, skb, 1, gfp);
  }
}
```

```c
static int tcp_transmit_skb(...) {
  // 克隆新skb出来
  if (likely(clone_it)) {
    skb = skb_clone(skb, gfp_mask);
  }

  // 封装TCP头

  th = tcp_hrd(skb);
  th->source = ...;
  ...
  // 调用网络层接口发送
  ...
}
```

1. `skb`必须复制，TCP重传机制。
2. `skb`包含了网络协议所有的头，通过偏移修改，节约内存。

### 4.3.3 网络层发送处理

忽略，处理路由表和ARP协议。

### 4.3.4 网络设备子系统

最关键的问题在于设备如何从RingBuffer中取skb:

```c
int quota = weight_p;

// 从队列中取出一个skb
while (qdisc_restart(q)) {
  // quota用尽
  // 其他进程需要CPU
  if (--quota <= 0 || need_resched()) {
    // 将触发一次NET_TX_SOFTIRQ类型的软中断
    _netif_schedule(q);
  }
}
```

### 4.3.5 后续

与接收类似，网卡驱动发送数据。

## 4.4 RingBuffer内存回收

发送完成后，CPU硬中断调用`NET_RX_SOFTIRQ`回收内存。

## 4.5 总结

### 内存拷贝

1. 数据从用户态到内核态发生的拷贝。
2. 为了保证可靠传输，skb必须进行拷贝，浅拷贝，内存损失不大。
3. 网络层时，当skb大于MTU时，必须申请额外的skb进行切片。
