# Chapter 4

Every container has something called a MNT *namespace* and a
unique file tree root.

There are three most common types of storage mounted
into containers:

+ Bind mounts
+ In-memory storage
+ Docker volumes

![Example of common container storage mounts](https://i.loli.net/2021/11/24/zjNabu9dcRynAvT.png)

All three types of mount points can be created using the
`--mount` flag on the `docker run` and `docker create`
subcommands.

## Bind mounts

*Bind mounts* are mount points used to remount parts of
a filesystem tree onto other locations.

The first problem with bind mounts is that they tie otherwise portable
container descriptions to the filesystem of a specific
host.

The next big problem is that they create an opportunity for
conflict with other containers.

## In-memory storage

Most service software and web applications use private
key files,
database passwords, API key files, and need upload buffering
space. In these cases, it is important that you never include those
types of files in an image or write them to disk.
Instead, you should use in-memory storage.

## Docker volumes

*Docker volumes* are named filesystem *trees* managed by Docker.
They can be implemented with disk storage on the host
filesystem,
or another more exotic backend such as cloud storage. All operations
on Docker volumes can be accomplished using the `docker volume`
subcommand set.

Semantically, a *volume* is a tool for for segmenting and sharing
data that has a scope or life cycle that's independent of a single
container. That makes volumes an important part of any containerized system design.
