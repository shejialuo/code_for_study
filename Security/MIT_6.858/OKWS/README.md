# Building Secure High-Performance Web Services with OKWS

## Design

### Practical Security Guidelines

+ *Server processes should be chrooted*.
+ *Server processes should run as unprivileged users*.
+ *Server processes should have the minimal set of database access privileges*.
+ *A server architecture should separate independent functionality into independent processes*.

### OKWS Design

OKWS provides Web developers with a set of libraries and helper processes so they can
build Web services as independent, stand-alone processes, isolated almost entirely from
the file system.

+ `okld` launches custom-built services and relaunches them should they crash.
+ `okd` routes incoming requests to appropriate Web services.
+ `pubd` provides Web services with limited read access to configuration files and HTML
template files store on the local disk.
+ `oklogd` writes log entries to disk.

## Implementation

To use OKWS, an administrator installs the helper binaries to a standard directory, and
installs the site specific services to a runtime jail directory. The administrator should
allocate two new UID/GID pairs for `okd` and `oklogd`.

### okld

The root process in the OKWS system is `okld`, the launcher daemon:

1. `okld` requests a new Unix socket connection from `oklogd`.
2. `okld` opens 2 socket pairs; one for HTTP connection forwarding, and for
one RPC control messages.
3. `okld` calls `fork`.
4. In the child address space, `okld` picks fresh UID/GID pairs $(x,x)$, sets the new
process's group list to $\{x\}$ and its UUID to $x$. It then changes directories into
`/cores/x`.
5. Still in the child address space, `okld` calls `execve`, launching the Web service.
The new Web service process inherits three file descriptors: one for receiving forwarding
HTTP connections, one for receiving RPC control messages, and one for RPC-based request
logging.
6. In the parent address space, `okld` sends the server side of the sockets opened in Step
2 to `okd`.

### okd

The `okd` process accepts incoming HTTP requests and demultiplexes them based on the
"Request-URI" in their first lines.
