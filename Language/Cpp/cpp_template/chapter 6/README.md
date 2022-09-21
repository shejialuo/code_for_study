# Chapter 6

## Perfect Forwarding

Suppose you want to write generic code that forwards the basic property of passed
arguments:

+ Modifyable object should be forwarded so that they still can be modified.
+ Constant objects should be forwarded as read-only objects.
+ Movable objects should be forwarded as movable objects.

