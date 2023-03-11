# 第六章 深入理解TCP连接建立过程

## 6.1 深入理解listen

### 6.1.1 listen系统调用

```c
SYSCALL_DEFINE2(listen, int, fd, int, backlog) {
  sock = sockfd_lookup_light(fd, &err, &fput_needed);
  if (sock) {
    // 获取内核参数 net.core.somaxconn
    somaxconn = sock_net(sock->sk)->core.sysctl_somaxconn;

    if ((unsigned int)backlog > somaxconn) {
      baclog = somaxconn;
    }

    // 调用协议栈注册的listen函数
    err = sock->ops->listen(sock, backlog);
  }
}
```

虽然`listen`允许我们传入`backlog`，但是如果用户传入的值比`net.core.somaxconn`还大的话是不会起作用的。

### 6.1.2 协议栈listen

```c
int inet_listen(struct sock *sock, int backlog) {
  if (old_state != TCP_LISTEN) {
    err = inet_csk_listen_start(sk, backlog);
  }

  // 设置全连接队列长度
  sk->sk_max_ack_backlog = backlog;
}
```

服务端的全连接队列长度是执行`listen`函数时传入的`backlog`和`net.core.somaxconn`的最小值。

```c
int inet_csk_listen_start(struct sock *sk, const int nr_table_entries) {
  struct inet_connection_sock *icsk = innet_csk(sk);

  // 接收队列内核对象的申请和初始化
  int rc = reqsk_queue_alloc(&icsk->icsk_accept_queue, nr_table_entries);
}
```

### 6.1.3 接收队列定义

`icsk->icsk_accept_queue`定义在`inet_connection_sock`下，是内核用来接收客户端请求的主要数据结构。全连接队列和半连接队列均在该数据结构中实现。

```c
struct inet_connection_sock {
  struct inet_sock icsk_inet;
  struct request_sock_queue icsk_accept_queue;
};
```

```c
struct request_sock_queue {
  // 全连接队列
  struct request_sock *rskq_accept_head;
  struct request_sock *rskq_accept_tail;

  // 半连接队列
  struct listen_sock *listen_opt;
}
```

对于全连接队列来说，其只需要先进先出即可。不需要查找。而对于半连接队列来说，需要找到第三次握手快速查找第一次握手时留存的`request_sock`对象，故使用一个哈希表来管理。

```c
struct listen_sock {
  u8 max_qlen_log;
  u32 nr_table_entries;
  ...
  struct request_sock *syn_table[0];
};
```

### 6.1.4 接收队列申请和初始化

申请和初始化都是常规工作，唯一复杂的在于半连接队列的计算，我猜想此处应该有论文进行证明。略。

### 6.1.5 总结

`listen`最主要的作用就是申请和初始化接收队列，包括全连接队列和半连接队列。

+ 全连接队列的长度: `min(backlog, net.core.somaxconn)`
+ 半连接队列的长度： 与`backlog`, `somaxconn`和`tcp_max_syn_backlog`都有关。

## 6.2 深入理解Connect

### 6.2.1 connect调用链展开

当在客户端机调用`connect`函数的时候，其通过系统调用调用sock的`connect`函数：

```c
int inet_stream_connect(struct socket *sock, ...) {
  ...
  __inet_stream_connect(sock, uaddr, addr_len, flags);
}

int __inet_stream_connect(struct socket *sock, ...) {
  struct sock *sk = sock->sk;

  switch (sock->state) {
    case SS_UNCONNECTED:
      err = sk->sk_port->connect(sk, uaddr, addr_len);
      sock->state = SS_CONNECTING;
      break;
  }
}
```

然后通过协议栈处理：

```c
int tcp_v4_connect(struct sock *sk, struct sockaddr *uaddr, int addr_len) {
  tcp_set_state(sk, TCP_SYN_SENT);

  // 动态选择一个端口
  err = inet_hash_connect(&tcp_death_row, sk);

  // 发送SYN
  err = tcp_connect(sk);

}
```

### 6.2.2 选择可用端口

```c
int inet_hash_connect(...) {
  return __inet_hash_connect(death_row, sk, inet_sk_port_offset(sk),
                             __inet_check_established, __inet_hash_nolisten);
}
```

+ `inet_sk_port_offset(sk)`：生成一个随机数。
+ `__inet_check_established`：检查是否和现有的ESTABLISH状态的连接冲突的时候用的函数。

```c
int __inet_hash_connect(...) {
  const unsigned short snum = inet_sk(sk)->inet_num;

  // 获取配置
  inet_get_local_port_range(&low, &high);
  remaining = (high - low) + 1;
  if (!snum) {
    for (int i = 1 ; i < remaining; i++) {
      port = low + (i + offset) % remaining;
    }
  }
}
```

在上述代码中，`inet_get_local_port_range`读取`net.ipv4.ip_local_port_range`内核参数确定端口号的范围。然后进入for循环，判断端口是否被占用，如果被占用，再去判断是否建立起了连接。

### 6.2.4 发起syn请求

+ 申请一个skb包，并将其设置为SYN包。
+ 添加到发送队列上。
+ 调用`tcp_transmit_skb`发送
+ 启动重传计时器。

### 6.2.5 小结

`connect`函数，设置本地socket的状态为`TCP_SYN_SENT`，选择一个可用的端口，发出SYN握手请求并启动重传计时器。

## 6.3 完整TCP连接

### 6.3.1 客户端connect

见6.2小节。

### 6.3.2 服务端响应SYN

在服务端，在接收过程中，所有的包都会进入`tcp_ipv4_rcv`，对于处于listen状态的socket，然后继续进入`tcp_v4_do_rcv`处理握手过程。

```c
int tcp_v4_do_rcv(struct sock *sk, struct sk_buff *skb) {
  // 服务端收到第一次握手SYN或者第三步ACK都会走到这里
  if (sk->state == TCP_LISTEN) {
    struct sock *nsk = tcp_v4_hnd_req(sk, skb);
  }
  ...
}
```

在`tcp_v4_do_rcv`中判断当前socket是listen状态后，首先会到`tcp_v4_hnd_req`查看半连接队列。服务端第一次响应SYN的时候，半连接队列必然空空如也。

```c
static sruct sock *tcp_v4_hnd_req(...) {
  // 查找半连接队列

  struct request_sock *req = inet_csk_search_req(...);

  ...
  return sk;
}
```

在`tcp_rcv_state_process`里面根据不同的`socket`状态进行不同的处理：

```c
int tcp_rcv_state_process(...) {
  switch (sk->sk_state) {
    case TCP_LISTEN:
      if (th->syn) {
        if (icsk->icsk_af_ops->conn_request(sk, skb) < 0) {
          return 1;
        }
      }
  }
}
```

`conn_request`是一个函数指针，指向`tcp_v4_conn_request`:

```c
int tcp_v4_conn_request() {
  if (inet_csk_request_queue_is_full(sk) && !isn) {
    want_cookie = tcp_syn_flood_action(sk, skb, "TCP");
    if (!want_cookie) {
      goto drop;
    }
  }

  if (sk_acceptq_is_full(sk) && inet_csk_reqsk_queue_young(sk) > 1) {
    ...
    goto drop;
  }

  // 分配 request_sock 内核对象
  req = inet_reqsk_alloc(&tcp_request_sock_ops);

  // 构造 syn + ack 包
  skb_synack = tcp_make_synack(...);


  if (likely(!do_fastopen)) {
    // 发送
    err = ip_build_and_send_ptk(...);

    // 添加到半连接队列，并开启计时器
    inet_csk_reqsk_queue_hash_add(...);
  }

}
```

+ 如果半连接队列满了，且未开启tcp_syncookies，该包直接被丢弃。
+ 如果全连接队列满了，且young_ack数量大于1，直接丢弃。

### 6.3.3 客户端响应SYNACK

清除`connect`时设置的重传定时器，把当前socket状态设置为ESTABLISHED，开启保活计时器后发出第三次握手的ACK确认。

### 6.3.4 服务端响应ACK

+ 去半连接队列查找`request_sock`，创建子`socket`。将其移除半连接队列，添加子`socket`到全连接队列。
+ 创建子`socket`仍然要判断全连接队列是否满了，如果满了，直接丢弃。
+ 设置状态为`ESTABLISHED`。

### 6.3.5 服务端accept

直接返回全连接队列的头元素即可。
