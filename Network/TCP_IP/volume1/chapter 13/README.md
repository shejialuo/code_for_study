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

<!-- TODO: add the picture -->
![A normal TCP connection establishment and termination](.)

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
