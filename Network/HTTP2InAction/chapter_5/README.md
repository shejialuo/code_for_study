# Chapter 5 Implementing HTTP/2 push

## 5.1 What is HTTP/2 server push?

HTTP/2 server push allows servers to send back extra resources that weren't
requested by the client. When using HTTP/1.1, if the page needed extra
resources to be displayed, the browser had to download the initial page, see
that extra resources were referenced, and then request them. Some resources
are critical to page rendering, and the browser won't even attempt to render
the page until these resources are downloaded. This process adds at least
one extra round trip, so it slows web browsing.

This round-trip delay led to performance optimizations such as inlining style
sheets directly onto the HTML page with `<style>` tags and doing something
similar in JavaScript with `<script>` tags.

HTTP/2 push breaks the "one request = one response" paradigm that HTTP has
always worked under. It allows the server to respond to one request with many
responses.

## 5.2 How to push

### 5.2.1 Using HTTP link headers to push

Many web servers and some CDNs use *HTTP link headers* to notify the web server
to push. If the web servers sees these HTTP headers, it pushes the resources
referenced in the header.

### 5.2.2 Pushing from downstream systems by using link headers

If you're using HTTP link headers to indicate resources to be pushed, these
headers don't need to be set in the web-server configuration.

## 5.3 How HTTP/2 push works in the browser

Instead of being pushed straight to a web page, a resource is pushed to a cache.
The web page is processed as it is normally. When the page sees the resource that
it needs, it checks the cache, finds it there, and loads it from the cache
rather than requesting it from the server.

### 5.3.1 Seeing how the push cache works

Pushed resources are held in a separate bit of memory (the HTTP/2 push cache)
waiting for the browser to request them, at which point they're loaded into the page;
if the caching headers are set, they're also saved in the browser's HTTP cache
as usual for later reuse.

The *push cache* isn't the first place where the browser looks for a resource.
Although the process is browser-specific, experiments show that if a resource
is available in the usual *HTTP cache*, the browser won't use the pushed resource.

![Browser interaction with HTTP2 push](https://s2.loli.net/2023/07/27/rXDfpu6hZlqoykP.png)

Following is a brief explanation of each cache:

+ The *image cache* is a short-lived, in-memory cache for that page the prevents
the browser from fetching an image twice. When the user browses away from
the page, the cache is destroyed.
+ The *preload cache* is another short-lived, in-memory cache used to hold
preloaded resources. Again, this cache is page-specific.
+ *Service workers* are fairly new background applications that run independently
of a web page and act as go-betweens for the web page and the website.
+ The *HTTP cache* is the main cache that most developers know about and
is a disk-based persistent cache shared across the browser, with a limited
size to be used for all the domains.
+ The *HTTP/2 push cache* is a short-lived, in-memory cache that is bound
to the connection and is checked last.

### 5.3.2 Refusing pushes with RST_STREAM

A client can refuse a pushed resource by sending an `RST_STREAM` frame on the
push stream with a `CANCEL` or `REFUSED_STREAM` code.
