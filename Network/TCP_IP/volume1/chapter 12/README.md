# Chapter 12 TCP: The Transmission Control Protocol (Preliminaries)

## 12.1 Introduction

*Information theory* indicates that the fundamental limits on the amount
of the information that can be moved across an information channel that is
*lossy*. There are two ways to solve this problem:

1. error-correcting codes
2. try sending again (*Automatic Repeat Request (ARQ)*)

### 12.1.1 ARQ and Retransmission

Instead of bit errors, there are many problems:

1. packet reordering
2. packet duplication
3. packet erasures (drop)

A straightforward method of dealing with packet drops (and bit errors) is to
resend the packet until it is received properly. This requires a way to determine:

+ whether the receiver has received the packet.
+ whether the packet it received was the same one the sender sent.

The method for a receiver to signal to a sender that it has received a packet is
called an *acknowledgement*, or ack. In its most basic form, the sender sends a
packet and awaits an ACK. When the receiver receives the packet, it sends the ACK.
However, there are some questions:

1. How long should the sender wait for an ACK.
2. What if the ACK is lost?
3. What if the packet was received but had errors in it.

For 1, it is complicated, the answer is in chapter 14. If an ACK is lost, the sender
just simply sends the packet again. For 3, we can use section 12.1 technologies.

There is the possibility that the receiver might receive *duplicate* copies of the
packet being transferred. This problem is addressed using a *sequence number*. Every
packet gets a new sequence number when it is sent at the source, and this sequence number
is carried along in the packet itself. The receiver can use this number to determine
whether it has already seen the packet and if so, discard it.

### 12.1.2 Windows of Packets

We define a *window* of packets as the collection of packets that have been injected by
the sender but not yet completely acknowledged. We refer to the *window size* as the number
of packets in the window.

### 12.1.3 Variable Windows: Flow Control and Congestion Control

For flow control, the window size is not fixed but is allowed to vary over time. The window
update and ACK are carried in a single packet, meaning the sender tends to adjust the size
of its window at the same time it slides it to the right.

Congestion control involves the sender slowing down so as to not overwhelm the network
between itself and the receiver.

### 12.1.4 Setting the Retransmission Timeout

A better strategy is to have the protocol implementation try to estimate them. This is
called *round-trip-time estimation* and is a statistical process.

## 12.2 TCP Header and Encapsulation

![The TCP header](https://s2.loli.net/2023/02/09/gJslfp5C2tjnHS6.png)

+ **CWR**: Congestion Window Reduced (the sender reduced its sending rate)
+ **ECE**: ECN Echo (the sender received an earlier congestion notification)
+ **URG**: Urgent (the *Urgent Pointer* field is valid, rarely used).
+ **ACK**: Acknowledgment (the *Acknowledgement Number* field is valid)
+ **PUSH**: Push (the receiver should put this data to the application as
soon as possible)
+ **RST**: Reset the connection.
+ **SYN** Synchronize sequence numbers to initiate a connection.
+ **FIN**: The sender of the segment is finished sending data to its peer.
