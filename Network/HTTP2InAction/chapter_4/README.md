# Chapter 4 HTTP/2 protocol basics

## 4.1 Why HTTP/2 instead of HTTP/1.2

The new version of the protocol differs by adding the following concepts:

+ Binary rather than textual protocol
+ Multiplexed rather than synchronous
+ Flow control
+ Stream prioritization
+ Header compression
+ Sever push

### 4.1.1 Binary rather than textual

HTTP/2 moves to a full binary protocol, in which HTTP messages are split and sent in
clearly defined frames. All HTTP/2 messages effectively use chunked encoding
as standard.

These frames are similar to the TCP packets that underlie most HTTP connections.
When all the frames are received, the full HTTP message can be reconstructed.

### 4.1.2 Multiplexed rather than synchronous

HTTP/2 allows multiple requests to be in progress at the same time, on a single
connection, using different streams for each HTTP request or response. This concept
of multiple independent requests happening at the same time was mde possible by
moving to the binary framing layer, where each frame has stream identifier.
The receiving party can construct the full message when all frames for that
stream have been received.

Frames are the key to allowing multiple messages to be sent at the same time.
Each frame is labeled to indicate which message it belongs to, which allows
two, three, or a hundred messages to be sent or received at the same time on the
same multiplexed connection.

![Requesting three resources across a multiplexed HTTP/2 connection](https://s2.loli.net/2023/07/26/Bcek5PYrMhKtpLJ.png)

This example shows that requests aren't sent at exactly the same time, each frame
needs to be sent after another on the same HTTP/TCP connection.

Similarly, resources can be sent back intermingled or sequentially. The order in which
the server sends back responses is entirely up to the server, though the client
can indicate priorities.

To prevent a clash of stream IDs client-initiated requests are given odd stream IDs, and
server-initiated requests are given even stream IDs. Stream ID 0 is a control stream
used by both client and server to manage the connection.

And we can get the following summaries:

+ HTTP/2 uses multiple binary frames to send HTTP requests and responses
across a single TCP connection, using a multiplexed stream.
+ HTTP/2 is mostly different at the message-sending level, and at even a slightly higher
level, the core concepts of HTTP remain the same.

### 4.1.3 Stream prioritization and flow control

Stream prioritization is implemented by the server sending more frames for
higher-priority requests than for lower-priority requests when a queue of frames
is waiting to be sent.

Flow control is another necessary consequence of using multiple streams over
the same connection. If the receiver is unable to process the incoming messages
as fast as the sender is sending, a backlog exists, which must be buffered and
eventually leads to packets being dropped and needing to be resent.

### 4.1.4 Header compression

HTTP headers are used to send additional information about requests and responses
from client to server, and vice versa. There's a lot of repetition in these headers,
as they're often sent identically for every resource.

### 4.1.5 Server push

Another important difference between HTTP/1 and HTTP/2 is that HTTP/2 adds
the concept of *server push*, which allows the server to respond to a request with
more than one response.

## 4.2 How an HTTP/2 connection is established

The HTTP/2 specification provides three ways to create the HTTP/2 connection:

+ Use HTTPS negotiation
+ Use the HTTP `Upgrade` header.
+ Use prior knowledge.

In theory, HTTP/2 is available over unencrypted HTTP, in which it is known
as h2c, and over encrypted HTTPS, in which it is known as h2. In practice, all
web browsers support HTTP/2 only over HTTPS (h2), so option 1 is used to
negotiate HTTP/2 by web browsers. Server-to-server HTTP/2 communication can be
over unencrypted HTTP (h2c) or HTTPS (h2), so it can use all these methods,
depending on which scheme is used.

### 4.2.1 Using HTTPS negotiation

#### HTTPS handshake

It's not difficult to understand here, because I have learned a lot in
computer security.

#### Application-layer protocol negotiation

ALPN added an additional extension to the `ClientHello` message， where clients
could advertise application protocol support, and also to the `ServerHello`
message, where servers could confirm which application protocol to use
after HTTPS negotiation.

![HTTPS handshake with ALPN](https://s2.loli.net/2023/07/26/WrioTHdYecJz3bF.png)

ALPN is simple and can be used to agree whether or not to use HTTP/2 for the
existing HTTPS negotiation messages without adding any further round trips, redirects,
or other upgrade delays.

#### Next protocol negotiation

NPN, worked in a similar manner. The main difference is that with NPN, the client
decides the protocol, whereas with ALPN, the server decides.

With NPN, the `ClientHello` message declares that the client is happy to
use NPN, the `ServerHello` message includes all the NPN protocols supported
by the server, and after encryption is enabled, the client picks the NPN protocol and
sends another message with this choice.

### 4.2.2 Using the HTTP upgrade header

A client can request to upgrade an existing HTTP/1.1 HTTP connection to HTTP/2
by sending an `Upgrade` HTTP header. This header should be used only for
unencrypted HTTP connections (h2c).

When a client sends an `Upgrade` header is entirely up to the client. The header
could be sent with every request, with the initial request only, or only if
the server has advertised HTTP/2 support via the `Upgrade` header in an HTTP response.

### 4.2.3 The HTTP/2 preface message

The first message that must be sent on an HTTP/2 connection is the HTTP/2 connection preface,
or "magic" string. This message is sent by the client as the first message
on the HTTP/2 connection.

```txt
RPI * HTTP/2.\r\n\r\nSM\r\n\r\n
```

The intention of this nonsensical, HTTP/1-like message is for when a client
tries to speak HTTP/2 to a server that doesn't understand HTTP/2. Such a
server tries to parse this message as it would any other HTTP message and fails
because it doesn't recognize the nonsense method or the HTTP version, and
it should reject the message. The sever, which knows that the client
speaks HTTP/2 based on the incoming message, doesn't send this magic
message; it must send a `SETTINGS` frame as its first message.
