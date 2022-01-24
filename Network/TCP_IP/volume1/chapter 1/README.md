# Chapter 1 Introduction

## 1.1 Architectural Principles

The Internet architecture should be able to interconnect multiple distinct networks
and that multiple activities should be able to run simultaneously on the resulting
interconnected network. Beyond this goal, there are some second-level goals:

+ Internet communication must continue despite loss of networks or gateways.
+ The Internet must support multiple types of communication services.
+ The Internet architecture must accommodates a variety of networks.
+ The Internet architecture must permit distributed management of its
resources.
+ The Internet architecture must be cost-effective.
+ The Internet architecture must permit host attachment with a low
level of effort.
+ The resources used in the Internet architecture must be accountable.

### 1.1.1 Packets, Connections, and Datagrams

In packet switching, "chunks" (packets) of digital information comprising
some number of bytes are carried through the network somewhat independently.
Chunks coming from different sources or senders can be mixed together and pulled after later,
which is called *multiplexing*.

When packages are received at a package switch, they are ordinarily stored in
*buffer memory* or *queue* and processed in a FIFO(FCFS) fashion
