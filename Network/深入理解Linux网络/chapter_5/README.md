# 第五章 深度理解本机I/O网络

## 5.1 本机发送过程

### 5.1.1 网络层路由

发送数据进入到协议栈到达网络层的时候，网络层入口函数是`ip_queue_xmit`。在网络层中会进行路由选择，路由选择完毕，再设置一些IP头，进行一些netfilter的过滤，将包交给邻居子系统。

对于本机网络I/O来说，特殊之处在于local路由表中就能找到路由项，对应的设备都将使用loopback网卡，也就是lo设备。其逻辑在`fib_lookup`函数体现。

```c
static inline int fib_lookup(...) {
  struct fib_table *table;

  table = fib_get_table(net, RT_TABLE_LOCAL);
  if (...) {

  }

  table = fib_get_table(net, RT_TABLE_MAIN);
  if (...) {

  }
}
```

我们可以执行`ip route list table local`和`ip route list table main`来查看路由表。

本机网络I/O仍然需要进行IP分片，但是lo设备的MTU设置的相当大。

### 5.1.2 本机IP路由

内核在初始化local路由表的时候，把local路由表里所有的路由项都设置成了`RTN_LOCAL`，不只是127.0.0.1。.

### 5.1.3 网络设备子系统

直接进入`dev_hard_start_xmit`进入驱动设备发送。

### 5.1.4 “驱动”设备

```c
static netdev_tx_t loopback_xmit(struct sk_buff *skb, struct net_device *dev) {
  // 剥离掉和原Socket的联系
  skb_orphan(skb);

  // 调用netif_rx
  if (likely(netif_rx(skb) == NET_RX_SUCCESS)) {}
}
```

在本机网络IO发送的过程中，传输层下面的skb就不需要释放了，直接给接收方发送过去，节省了一点点开销。
然后仍然触发软中断，发送完成。

## 5.2 本机接收过程

在本机网络IO的接收过程，由于并不真正地经过网卡，所以网卡的发送过程、硬中断全部省去，直接从软中断开始。
