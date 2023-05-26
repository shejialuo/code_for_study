# Chapter 14 TCP Timeout and Retransmission

## 14.1 Introduction

When data segments or acknowledgments are lost, TCP initiates a *retransmission*
of the data that has not been acknowledgements. TCP has two separate mechanisms
for accomplishing retransmission, one based on time and one based on the
structure of the acknowledgments. The second approach is usually much more
efficient than the first.

TCP sets a timer when it sends data, and if the data is not acknowledged when
the timer expires, a *timeout* or *time-based retransmission* of data occurs.
THe timeout occurs after the interval called the *retransmission timeout*(RTO).
It has another way of initiating a retransmission called *fast retransmission*
or *fast retransmit*.

When the sender believes that the receiver might be missing some data, a choice
needs to be made between sending new unsent data and retransmitting.

## 14.2 Simple Timeout

The doubling of time between successive retransmissions is called a
*binary exponential backoff*.

Logically, TCP has two thresholds to determine how persistently it will attempt
to resend the same segment:

+ Threshold R1 indicates the number of tries TCP will make to resend a segment
before passing "negative device" to the IP layer.
+ Threshold R2 dictates the point at which TCP should abandon the connection.

In Linux, the R1 and R2 values for regular data segments are available to be
changed by applications or can be changed using the system-wide configuration
variables `net.ipv4.tcp_retries1` and `net.ipv4.tcp_retries2`, respectively.

However, TCP needs to dynamically determine its RTO.

## 14.3 Setting the RTO

Fundamental to TCP's timeout and retransmission procedures is how to set the
RTO based upon measurement of the RTT experienced on a given connection. If
TCP retransmits a segment earlier than the RTT, it may injecting duplicate
traffic into the network unnecessarily. If it delays sending until much longer
than the RTT, the overall network utilization drops when traffic is lost.

Because TCP sends acknowledgements when it receives data, it is possible to
send a byte with a particular sequence number and measure the time required
to receive an acknowledgement that covers that sequence number. Each such
measurement is called an RTT *sample*.

### 14.3.1 The Classic Method

The original TCP specification had TCP update a *smoothed RTT estimator*(SRTT)
using the following formula:

$$
SRTT \leftarrow \alpha(SRTT) + (1 - \alpha)RTT_{s}
$$
