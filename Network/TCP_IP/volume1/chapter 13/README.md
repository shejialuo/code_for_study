# Chapter 13 TCP Connection Management

## 13.1 Introduction

TCP is a unicast *connection-oriented* protocol. Before either end can
send data to the other, a connection must be established between them.

Because of its management of *connection state*, TCP is a considerably
more complicated protocol than UDP.

During connection establishment, several *options* can be exchanged
between the two endpoints regarding the parameters of the connection.
Some options are allowed to be sent only when the connection is established,
and others can be sent later.

## 13.2 TCP Connection Establishment and Termination

A connection typically goes through three phases: setup, data transfer
(called established), and teardown (closing). A typical TCP connection
establishment and close is shown below

![A normal TCP connection establishment and termination](https://s2.loli.net/2023/02/13/vBUFHVPYzWCEyMe.png)

The figure shows a timeline of what happens during connection establishment.
To establish a TCP connection, the following events usually take place:

1. The *active opener* sends a SYN segment specifying the port number of the
peer to which it wants to connect and the client's initial sequence number
(`ISN(c)`). It typically sends one or more options at this point.
2. The server responds with its own SYN segment containing its initial
sequence number (`ISN(s)`). The server also acknowledges the client's SYN
by ACKing `ISN(c)` plus 1. A SYN consumes one sequence number and is
retransmitted if lost.
3. The client must acknowledge the SYN from the server by ACKing `ISN(s)`
plus 1.

These three segments complete the connection establishment. This is often
called the *three-way handshake*.

The figure also shows how a TCP connection is closed. Either end can
initiate a close operation, and simultaneous closes are also supported but
are rare. The closing TCP initiates the close operation by sending a FIN
segment. The complete close operation occurs after both sides have completed
the close:

1. The *active closer* sends a FIN segment specifying the current sequence
number the receiver expects to see (`K`). The FIN also includes an ACK for the
last data sent in the other direction.
2. The *passive closer* responds by ACKing value `K + 1` to indicate its
successful receipt of the active closer's FIN. At this point, the application
is notified that the other end of its connection has performed a close.
The passive closer then becomes another active closer and sends its own FIN.
The sequence number is equal to `L`.
3. To complete the close, the final segment contains an ACK for the last FIN.
Note that if a FIN is lost, it is retransmitted until an ACK for it is received.

### 13.2.1 TCP Half-Close

TCP supports a half-close operation. Few applications requires this capacity, so
it is not common. The Berkeley sockets API supports half-close, if the application
calls the `shutdown()` function.

### 13.2.2 Initial Sequence Number (ISN)

It might be possible to have TCP segments being routed through the network that
could show up later and disrupt a connection.

Before each end sends its SYN to establish the connection, it choose an ISN for
that connection. The ISN should change over time, so that each connection has
a different one. \[RFC0793\] specifies that the ISN should be viewed as a 32-bit
counter that increments by 1 every 4us. The purpose of doing this is to arrange
for the sequence numbers for segments on one connection to not overlap with
sequence numbers on a another identical connection.

If a connection had one of its segments delayed for a long period of time and
closed, but then opened again with the same 4-tuple, it is conceivable that the
delayed segment could reenter the new connection's data stream as valid data. This
would be most troublesome. However, it does suggest that an application with a
very great need for data integrity should *employ its own CRCs or checksums* at
the application layer.

### 13.2.3 Timeout of Connection Establishment

The number of times to retry an initial SYN can be configured on some systems and
usually has a fairly small value such as 5.

+ `net.ipv4.tcp_syn_retries`: maximum number of times to attempt to resend SYN
segment.
+ `net.ipv4.tcp_synack_retries`: maximum number of times to attempt to resend SYN+ACK
segment.

## 13.3 TCP State Transitions

### 13.3.1 TCP State Transition Diagram

Below shows the normal TCP connection establishment and termination, detailing
the different states through which the client and server pass.

![Tcp states corresponding to normal connection establishment and termination](https://s2.loli.net/2023/02/14/5qAzueJDbIEl18g.png)

### 13.3.2 TIME_WAIT (2MSL) State

The TIME_WAIT state is also called the 2MSL wait state. It is a state in which
TCP waits for a time equal to twice the *Maximum Segment Lifetime* sometimes called
*time_wait*.

This lets TCP resend the final ACK in case it is lost. The finial ACK is resent not
because the TCP retransmits ACKs, but because the other side will transmit its FIN.
Indeed, *TCP will always retransmit FINs until it receives a final ACK*.

Another effect of this 2MSL wait state is that while the TCP implementation waits,
the endpoints defining that connection cannot be reused. That connection can be
reused only when the 2MSL wait is over, or when a new connection uses an ISN that
exceeds the highest sequence number used on the previous instantiation of the
connection, or if the use of the Timestamps option allows the disambiguation of
segments from a previous connection instantiation to not be confused.

Most implementation and APIs provide a way to bypass this restriction. With the
Berkeley sockets API, the `SO_REUSEADDR` socket option enables the bypass option.
However, even with this bypass mechanism for one socket, the rules of TCP still
prevent this port number from being reused by another instantiation of the same
connection that is in the 2MSL wait state. Any delayed segments that arrive for
a connection while it is in the 2MSL wait state are discarded.

### 13.3.3 Quiet Time Concept

The 2MSL provides protection against delayed segments from an earlier instantiation
of a connection being interpreted. But this works only if a host with connections
in the 2MSL wait does not crash.

When a host crashed, the 2MSL can't provide such protection. So TCP should wait an
amount of time equal to the MSL before creating any new connections after a reboot
or crash. This is called the *quiet time*. Few implementations abide by this because
most hosts take longer than the MSL to reboot after a crash.

### 13.3.4 FIN_WAIT_2 State

In the FIN_WAIT_2 state, the TCP must wait for the application on the other end
to recognize that it has received an end-of-file notification and close its
end of the connection, which causes a FIN to be sent. However, this might be
endless.

Many implementations prevent this infinite wait in the FIN_WAIT_2 state as follows:
A timer would be set. If the connection is idle when the timer expires. TCP moves
the connection into the CLOSED state.

## 13.4 Reset Segments

For a reset segment to be accepted by a TCP, the ACK bit field must be set and
the ACK Number field must be within the valid window.

### 13.4.1 Aborting a Connection

+ *Orderly release*: terminate a connection normally.
+ *Abortive release*: abort a connection by sending a reset.

Aborting a connection provides two features to the application:

1. any queued data is thrown away and a reset segment is sent immediately.
2. the receiver of the reset can tell that the other end did an abort.

The sockets API provides this capability by using the "linger on close"
socket option (SO_LINGER) with a 0 linger value.

### 13.4.2 Half-Open Connections

A TCP connection is said to be *half-open* if one end has closed or aborted
the connection without the knowledge of the other end. This can happen anytime
one of the peers crashes.

Another common cause of a half-open connection is when one host is powered
off instead of shut down properly. This happens, when PCs are being used to
run remote login clients and are switched off at the end of the day. If there
was no data transfer going on when the power was cut, the server will *never*
know the client disappeared. When the user comes in the next morning, powers
on the PC, and starts a new session, a new occurrence of the server is started
on the server host. This can lead to many half-open TCP connections on the
server host.

### 13.4.3 TIME-WAIT Assassination

When client remains in the TIME_WAIT state, when client receives old
segments from server, it determines that both the sequence number
and ACK values are "old". When receiving such old segments, TCP
responds by sending an ACK with the most current sequence number
and ACK values. However, when the server receives this segment, it
has no information whatsoever about the connection and therefore
replies with an RST segment. It would cause the client to prematurely
transition from TIME_WAIT to CLOSED. Most systems avoid this problem
by simply not reacting to reset segments while in the TIME_WAIT state.
