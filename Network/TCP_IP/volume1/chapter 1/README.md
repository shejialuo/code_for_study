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
*buffer memory* or *queue* and processed in a FIFO(FCFS) fashion.

Alternative techniques, such as TDM and *static multiplexing* typically reserve a
certain amount of time or other resources for data on each connection. They may not
fully utilize the network capacity because reserved bandwidth may go unused.

### 1.1.2 The End-to-End Argument and Fate Sharing

When large systems such as an operating system or protocol suite are being designed,
a question often arises as to where a particular feature or function should be
placed. One of the most important principles that influenced the design of the
TCP/IP suite is called the *end-to-end* argument.

> The function in question can completely and correctly be implemented only with
> the knowledge and help of the application standing at the end points of the
> communication system.

*Fate sharing* suggests placing all the necessary state to maintain an active
communication association at the same location with the communicating endpoints.

### 1.1.3 Error Control and Flow Control

As an alternative to the overhead of reliable, in-order delivery implemented
within the network, a different type of service called *best-effort delivery*
was adopted by Frame Relay and the Internet Protocol.

## 1.2 Design and Implementation

We make a distinction between the protocol architecture and the *implementation architecture*,
which defines how the concepts in a protocol architecture may be rendered into
existence, usually in the form of the software.

### 1.2.1 layering

When layering, each layer is responsible for a different facet of the communications.

### 1.2.2 Multiplexing, Demultiplexing, and Encapsulation in Layered Implementations

One of the major benefits of a layered architecture is its natural ability to perform
*protocol multiplexing*. This form of multiplexing allows multiple different protocols
to coexist on the same infrastructure.

Multiplexing can occur at different layers, and at each layer a different sort of
*identifiers* is used for determining which protocol or stream of information belongs
together.

One other important feature of layering is that in pure layering not all worked devices
need to implement all the layers.
