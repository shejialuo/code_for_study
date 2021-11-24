# Chapter 7

## Configuring image attributes

When you use `docker container commit`, you commit a new layer
to an image. The filesystem snapshot isn't the only thing included
with this commit. Each layer also includes metadata describing
the execution context.

Of the parameters that can be set when a container is created, all
the following will carry forward with an image created from
the container:

+ All environment variables
+ The working directory
+ The set of exposed ports
+ All volume definitions
+ The container entrypoint
+ Command and arguments

## Union filesystems

The idea is similar to Git.
