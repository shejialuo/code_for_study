# Chapter 2 The road to HTTP/2

## 2.1 HTTP/1.1 and the current World Wide Web

### 2.1.1 HTTP/1.1's fundamental performance problem

Because HTTP/1.1 would handle a request one by one. The performance would be bad.

### 2.1.2 Pipelining for HTTP/1.1

Pipelining should have brought huge improvements to HTTP performance, but for
many reasons, it was difficult to implement, easy to break, and not well supported
by web browsers or web servers.

Even if pipelining were better supported, it still requires responses to be returned
in order in which they were requested. This problem is known as *head-of-line blocking*.

## 2.2 Workarounds for HTTP/1.1 performance issues

There are two categories to overcome the performance limitations of HTTP/1.1:

+ Use multiple HTTP connections.
+ Make fewer but potentially larger HTTP requests.

### 2.2.1 Use multiple HTTP connections

One of the easiest ways to get around the blocking issue of HTTP/1.1 is to open
more connections, allowing parallelization to have multiple HTTP requests.
Most browsers open six connections per domain for this reason.

To increase this limit of six further, many websites serve static assets from
subdomains, allowing web browsers to open a further six connections for each
new domain. This technique is known as *domain sharing*.

However, tha main issue with multiple HTTP connections, however, is significant
inefficiencies with the underlying TCP protocol.

TCP starts cautiously, with a small number of packets sent before acknowledgement.
The congestion window gradually increases over time as the connection is shown
to be able to handle larger sizes without losing packets. The size of the TCP
congestion window is controller by the TCP *slow-start* algorithm. TCP packets
in the CWND must be acknowledged before more packets can be sent.

Therefore, with a small CWND, it may take several TCP acknowledgements to
send the full HTTP request messages.

### 2.2.2 Make fewer requests

The other common optimization technique is to make fewer requests, which involves
reducing unnecessary requests or requesting the same amount of data over fewer
HTTP requests. The former method involves using HTTP cache headers. The latter
method involves bundling assets into combined files.

For images, this bundling technique is known as *spriting*. If you have a lot of
social media icons on your website, you could use one file for each icon.
But this method would lead to a lot of inefficient HTTP queuing. Instead, you could bundle
them into one large image file and then use CSS to pull out sections of the image
to effectively re-create the individual images.

The main downside to this solution is the complexity it introduces. Another
downside it the waste in these files. The final issue is caching. If you
cache your sprite image for a long time but then need to add an image, you
have to make the browsers download the while file again.

## 2.3 Moving from HTTP/1.1 to HTTP/2

### 2.3.1 SPDY

SPDY was built on top of HTTP, but doesn't fundamentally change the protocol, in
much the same way that HTTPS wrapped around HTTP without changing its underlying
use.

SPDY introduces a few important concepts to deal with the limitations of HTTP/1.1:

+ *Multiplexed streams*: Requests and responses used a single TCP
connection and were
broken into interleaved packets grouped into separate streams.
+ *Request prioritization*: To avoid introducing new performance problems by
sending all requests at the same time, the concept of prioritization of
the requests was introduced.
+ *HTTP header compression*

### HTTP/2

The HTTP/2 is the next generation of the HTTP, and it is based on the SPDY.
