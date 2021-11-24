# Chapter 5

## Docker container networking

By default, Docker includes three networks, and each is provided by
a different driver.

+ The `bridge` driver provides intercontainer connectivity for all
containers running on the same machine.
+ The `host` driver instructs Docker not to create any
special
networking namespace or resources for attached containers.
+ The `none` network uses the `null` driver. Containers attached
to the `none` network will not have any network connectivity.

The **scope* of a network can take three values:

+ `local`
+ `global`
+ `swarm`

### Handling inbound traffic with NodePort publishing

Since container networks are connected to the broader network via
network address translation, you have to specifically tell Docker
how to forward traffic from the external network interfaces.

*NodePort publishing* is a term we've used here to match Docker
and other ecosystem projects. The *Node* portion is an interface
to the host as typically a node in a larger cluster of machines.

Port publication configuration is provided at container creation
time and cannot be changed later. The `docker run` and
`docker create` commands provide a `-p` or `--publish` list option.

All of the following arguments are equivalent:

+ `0.0.0.0:8080:8080/tcp`
+ `8080:8080/tcp`
+ `8080:8080`
