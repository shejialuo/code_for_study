# 第二章 内核是如何接收网络包的

## 2.1 数据是如何从网卡到协议栈的

### 2.1.1 Linux网络收包总览

内核和网络设备驱动都是通过中断的方式来处理的。当设备上有数据到达时，会给CPU的相关引脚发一个电压变化，以通知CPU来处理数据。Linux中断处理函数分上半部和下半部。上半部只进行最简单的工作，快速处理然后释放CPU。将剩下的绝大部分的工作都放到下半部。
2.4以后的Linux内核版本采用的下半部实现方式是*软中断*，由`ksoftirqd`内核线程全权处理。硬中断是通过CPU物理引脚施加电压变化实现的。

![内核收包路径](https://s2.loli.net/2022/11/12/vUkzacLNq9bHf6K.png)

当网卡收到数据以后，以DMA的方式把网卡收到的帧写到内存里，再向CPU发起一个中断，以通知CPU由数据到达。当CPU收到中断请求后，会去调用网络设备驱动注册的中断处理函数。网卡的中断处理函数并不做过多工作，发出软中断请求，然后尽快释放CPU资源。`ksoftirqd`内核线程检测到有软中断请求到达，调用`poll`开始轮询收包，收到后交给各级协议栈处理。

### 2.1.2 Linux启动

#### 创建ksoftirqd内核线程

内核会创建`ksoftirqd`线程。首先其在`kernel/softirq.c`定义了线程的结构：

```c
static struct smp_hotplug_thread softirq_threads = {
  .store = &ksoftirqd,
  .thread_should_run = ksoftirqd_should_run,
  .thread_fn = run_ksoftirqd,
  .thread_comm = "ksoftirqd/%u"
}
```

内核会调用`spawn_ksoftirdq`对每一个CPU注册线程。

```c
static __init int spawn_ksoftirqd(void) {
  register_cpu_notifier(&cpu_nfb);

  BUG_ON(smpboot_register_percpu_thread(&softirq_threads));
  return 0;
}
```

`smpboot_register_percpu_thread`位于`kernel/smoboot.c`中，直接创建线程。

```c
int smpboot_register_percpu_thread(struct smp_hotplug_thread *plug_thread) {
  unsigned int cpu;
  int ret = 0;

  mutex_lock(&smpboot_threads_lock);
  for_each_online_cpu(cpu) {
    ret = __smpboot_create_thread(plug_thread, cpu);
    if (ret) {
      smpboot_destroy_threads(plug_thread);
      goto out;
    }
    smpboot_unpark_thread(plug_thread, cpu);
  }
  list_add(&plug_thread->list, &hotplug_threads);
out:
  mutex_unlock(&smpboot_threads_lock);
  return ret;
}
```

#### 网络子系统初始化

对于每一个CPU，内核都会初始化`softnet_data`结构体：

```c
// include/linux/netdevice.h
struct softnet_data {
  struct Qdisc *output_queue;
  struct Qdisc **output_queue_tailp;
  struct list_head poll_list;
  struct sk_buff *completion_queue;
  struct sk_buff_head process_queue;

  /* states */
  ...

  unsigned int dropped;
  struct sk_buff_head input_pkt_queue;
  struct napi_struct  backlog;
}
```

内核通过`net_dev_init`实现网络系统的初始化，其为每一个CPU分配一个`softnet_data`结构体，该数据结构的`poll_list`用于等待驱动程序将其`poll`函数注册进来。`NET_TX_SOFTIRQ`的处理函数为`net_tx_action`，`NET_RX_SOFTIRQ`的处理函数为`net_rx_action`。

```c
// file: net/core/dev.c
static int __init net_dev_init(void) {
  ...
  for_each_possible_cpu(i) {
    struct softnet_data *sd = &per_cpu(softnet_data, i);

    //数据结构的初始化
  }

  open_softirq(NET_TX_SOFTIRQ, net_tx_action);
  open_softirq(NET_RX_SOFTIRQ, net_rx_action);
}
```

#### 协议栈初始化

协议栈是通过注册入内核实现的。其本质的思路是TCP协议和UDP协议注册到`inet_protos`数组中。同时`ptype_base`存储着`ip_rcv`函数的地址。

#### 网卡驱动初始化

每一个驱动程序会使用`module_init`向内核注册一个初始化函数，当驱动程序被加载时，内核会调用这个函数。

```c
//file: drivers/net/ethernet/intel/igb/igb_main.c

static int __init igb_init_module(void) {
  ...
  ret = pci_register_driver(&igb_driver);
  return ret;
}
```

驱动的`pci_register_driver`调用完成后，Linux内核就知道了该驱动的相关信息。当网卡设备被识别后，内核会调用其驱动的probe方法。驱动的probe方法执行的目的就是让设备处于ready状态。对于igb网卡，其执行的函数为`igb_probe`

网卡驱动实现了`ethtool`所需要的接口，也在这里完成函数地址的注册。当`ethtool`发起了一个系统调用之后，内核会找到对应操作的回调函数。

同时注册`net_device_ops`用的是`igb_netdev_ops`变量，其中包含`igb_open`等函数，该函数在网卡启动的时候会被调用

#### 启动网卡

启动网卡时，内核会调用`net_device_ops`的`ndo_open`方法，指向`igb_open`，其本质是做了一系列的工作，分配传输描述符数组、接收描述符数组、注册中断处理函数，启用NAPI。

```c
//file: drivers/net/ethernet/intel/igb/igb_main.c

static int __igb_open(struct net_device *netdev, bool resuming) {
  err = igb_setup_all_tx_resources(adapter);
  err = igb_setup_all_rx_resources(adapter);
  err = igb_request_irq(adapter);

  if (err)
    goto err_req_irq;

  for (i = 0; i < adapter->num_q_vectors; i++)
    napi_enable(&(adapter->q_vector[i]->napi));
}
```

其中`igb_setup_all_rx_resources`通过调用`igb_setup_rx_resources`对每个`struct igb_ring`进行初始化。

`igb_ring`的数据结构很容易理解，其核心的内容在于`igb_rx_buffer`：

```c
struct igb_rx_buffer {
  dma_addr_t dma;
  struct page *page;
  unsigned int page_offset;
}
```

可以看见每一个buffer对应了一个页表以及页偏移量。

`igb_setup_rx_resources`其首先在内核中给队列分配内存。然后初始化队列的基本元素。

由于采用DMA的方式，存在两个队列：

+ `igb_rx_buffer`：内核使用的队列
+ `e1000_adv_rx_desc`：硬件使用的数据结构

### 2.1.3 迎接数据的到来

#### 硬中断处理

当数据帧从网线到达网卡时，第一站是网卡的接收队列。网卡在分配给自己的RingBuffer中寻找可用的内存位置，找到后DMA引擎会把数据DMA到网卡之前关联的内存里。当DMA操作完成后，网卡会向CPU发起一个硬中断，通知CPU有数据到达。

网卡的硬中断注册的处理函数是`igb_msix_ring`:

```c
//file: drivers/ethernet/intel/igb/igb_main.c

static irqreturn_t igb_msix_ring(int irq, void *data)
{
  struct igb_q_vector *q_vector = data;

  /* Write the ITR value calculated from the previous interrupt. */
  igb_write_itr(q_vector);

  napi_schedule(&q_vector->napi);

  return IRQ_HANDLED;
}
```

关键在于`napi_schedule`其最终调用`————napi_schedule`，其做的工作很简单：添加`poll_list`到末尾，然后触发软中断。

```c
//file: net/core/dev.c
static inline void __napi_schedule(struct softnet_data *sd, struct napi_struct *napi) {
  list_add_tail(&napi->poll_list, *sd->poll_list);
  __raise_softirq_irqoff(NET_RX_SOFTIRQ);
}
```

软中断的处理过程很简单，基于`smp_processor_id`函数获取硬中断写下的信息如果某一位的值是1进行处理，然后移位。在上述中，网卡接收的软中断函数是`net_rx_action`，其核心的思路相当简单，取目前CPU的`softnet_data`，然后遍历`poll_list`，调用每个元素的`poll`方法。

`igb_poll`是我们关心的方法，其核心是调用`igb_clean_rx_irq`方法。在此之前，你应该了解socket buffer数据结构。其核心的思路也很简单，RingBuffer存储的对象就是一个又一个的socket buffer，硬件在收包的时候，就会将网络包转化为socket buffer放在RingBuffer中然后内核再一个一个地取。

然后数据包被送到协议栈处理。

#### 网络协议栈处理

`netif_receive_skb`函数会根据包的协议进行处理，其首先调用`__netif_receive_skb`函数然后再调用`__netif_receive_skb_core`函数。

接着`__netif_receive_skb_core`函数取出协议信息，送到根据已经注册在内核的协议栈处理函数。

### 2.1.4 收包小结

收包之前，内核需要做很多的开始工作：

+ 创建软中断处理线程`ksoftirqd`。
+ 协议栈注册。
+ 网卡驱动初始化，准备自己的DMA，将NAPI的`poll`告诉内核。
+ 启动网卡，初始化RX、TX队列。

当数据到来以后，第一个迎接的网卡：

+ 网卡将数据帧DMA到内存的RingBuffer中，存储的类型是socket buffer，然后发出硬中断。
+ CPU响应中断，调用中断处理函数，中断处理函数主要是置位软中断。
+ `ksoftirqd`处理软中断，首先关闭硬中断。
+ `ksoftirqd`调用驱动的`poll`函数。

## 2.2 本章总结

+ RingBuffer到底是什么，RingBuffer为什么会丢包？

RingBuffer本质就是内核的一块内存，其存储一个又一个的socket buffer数据结构。软中断收包时把socket buffer取走，并申请新的socket buffer放入。

当内核处理不及时导致RingBuffer满了，数据包就会被丢弃，从而丢包。

+ tcpdump是如何工作的？

tcpdump通过模拟一个虚拟设备来工作。

+ 网络接收过程中的CPU开销如何查看

在网络包的接收处理过程中，主要工作集中在硬中断和软中断上，二者的消耗都可以通过`top`命令来查看：`hi`是硬中断开销，`si`是软中断开销。

+ DPDK是什么

直接从网卡收数据的技术。
