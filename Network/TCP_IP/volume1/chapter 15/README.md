# Chapter 15 TCP Data Flow and Window Management

## 15.1 Interactive Communication

The most of all TCP segments contain *bulk data* and the remaining
portion contains *interactive data*. Bulk data segments tend to
be relatively large, while interactive data segments tend to be
much smaller.

TCP handles both types of data using the same protocol and packet
format, but different algorithms com into play for each.

Each interactive keystroke normally generates a separate data packet.
That is, the keystrokes are sent from the client to the server individually
(one character at a time rather than one line at a time).

A single typed character could thus generate four TCP segments (for `ssh`):

1. the interactive keystroke from the client.
2. an acknowledgement of the keystroke from the server.
3. the echo of the keystroke from the server.
4. an acknowledgement of the echo from the client back to the server.

Normally, segment 2 and segment 3 are combined.


