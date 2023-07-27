# Chapter 1 Web technologies and HTTP

## 1.1 The syntax and history of HTTP

### 1.1.1 HTTP/0.9

The first published specification for HTTP, was version 0.9, issued in 1991.
It specifies that a connection is made over TCP/IP to a server and optional port.
A single line of ASCII text should be sent, consisting of `GET`, the document
address (with no spaces), and a carriage return and line feed.

Following is the only possible command in HTTP/0.9:

```txt
GET /section/page.html/r/n
```

There was no concept of HTTP header fields or any other media.

### 1.1.2 HTTP/1.0

HTTP/1.0 adds some key features, including:

+ More request methods: `HEAD` and `POST` was added.
+ Addition of an optional HTTP version number for all messages.
+ HTTP headers, which could be sent with both the request and the response
to provide more information.
+ A three digit response code.

### 1.1.3 HTTP/1.1

Many of the additional features of HTTP/1.1 were introduced through HTTP
headers.

#### Mandatory host header

The URL provided by HTTP request lines isn't an absolute URL but a relative
URL. When HTTP was created, it was assumed that a web server would host only
one website. Nowadays, many web servers host several sites on the same
sever, so it's important to tell the server which site you want as well as
the which relative URL you want on that site. This feature was implemented
by adding a `Host` header in the request:

```txt
GET / HTTP/1.1
Host: www.google.com
```

#### Persistent connections (keep alive)

Initially, HTTP was a single request-and-response protocol, a client opens
the connection, requests a resource, gets the response and the connection
is closed.

This problem was resolved with a new `Connection` HTTP header that could be sent
with an HTTP/1.0 request. By specifying the value `Keep-Alive` in this header, the
client is asking the server to keep the connection open to allow the sending of
additional requests:

```txt
GET /page.html HTTP/1.0
Connection: Keep-Alive
```

The server would respond as usual, but if it supported persistent connections,
it included a `Connection: Keep-Alive` header in the response:

```txt
HTTP/1.0 200 OK
Date: Sun, 25 Jun 2017 13:30:24 GMT
Connection: Keep-Alive
Content-Type: text/html
Content-Length: 12345
Server: Apache
<!doctype html>
<html>
<head>
...etc.
```

This response tells the client that it can send another request on the same
connection as soon as the response is completed. The `Content-Length` HTTP
header must be used to define the length of the response body, and when the
entire body has been received, the client is free to send another request.

Any HTTP/1.1 connection should be assumed to be using persistent connections
even without the presence of the `Connection: Keep-Alive` header in the
response.

If the server did want to close the connection, it had to explicitly include
a `Connection: close` HTTP header in the response.

HTTP/1.1 added the concept of pipelining, so it should be possible to send
several requests over the same persistent connection and get the response
back in order.

#### Other new features

HTTP/1.1 introduced many other features:

+ Better cache methods. These methods allowed the server to instruct the client
to store the resource in the browser's cache so it could be reused later if required.
The `Cache-Control` HTTP header introduced in HTTP/1.1 had more options than
the `Expires` header from HTTP/1.0.
+ HTTP cookies to allow HTTP to move from being a stateless protocol.
+ The introduction of character sets and language in HTTP responses.
+ Proxy support.
+ Authentication.
+ New status codes.
+ Trailing headers.
