# Process Relationships

## Terminal Logins

Let's start by looking at the programs that are executed when we log in to a UNIX
system. In early UNIX systems, users logged in using dumb terminals that were
connected to the host with hard-wired connections. These logins came through a
terminal device driver in the kernel.

The system administrator creates a file, usually `/etc/ttys`, that has one line
per terminal device. Each Line specifies the name of the device and other
parameters that are passed to the `getty`. One parameter is the baud rate of
the terminal, for example.

When the system is bootstrapped, the kernel creates process ID 1, the `init` process,
and it is `init` that brings the system up in multiuser mode. The `init` process
reads the file `/etc/ttys` and, for every terminal device that allows a login,
does a `fork` followed by an `exec` of the program `getty`.

It is `getty` that calls `open` for the terminal device. Once the device is open,
file descriptors 0, 1, and 2 are set to the device. Then `getty` outputs something
like `login:` and waits for us to enter our name.

When we enter our user name, `getty`'s job is complete, and it then invokes
`login` program.

![Login process](https://i.loli.net/2021/10/10/W6xAFDYRgli329O.png)

The `login` program does many things. Since it has our user name, it can call
`getpwnam` to fetch our password file entry. Then `login` calls `getpass` to display
the prompt `Password:` and read our password. It calls `crypt` to encrypt the
password that we entered and compares.

If we log in correctly, `login` will

+ Change to our home directory (`chdir`)
+ Change the ownership of our terminal device (`chown`) so we own it.
+ Change the access permissions for our terminal device so we have permission to
  read from and write to it.
+ Set our group IDs by calling `setgid` and `initgroups`.
+ Initialize the environment with all the information that `login` has.
+ Change to our user ID and invoke our login shell.

## Network Logins

In the case of network logins, all the logins come through the kernel's network
interface drivers, and we don't know ahead of time how many of these will occur.

To allow the same software to process logins over both terminal logins and network
logins, a software driver called a *pseudo terminal* is used to emulate the behavior
of a serial terminal and map terminal operations to network operations, and
vice versa.

In Linux, a single process waits for most network connections: the `inetd` process,
soemtimes called the *Internet superserver*.

As part of the system start-up, `init` invokes a shell that executes the shell
script `/etc/rc`. One of the daemon that is started by this shell script is
`inetd`. Once the shell script terminates, the parent process of `inetd` becomes
`init`; `inetd` waits for TCP/IP connection requests to arrive at the host.

## Process Group

In addition to having a process ID, each process belongs to a process group. A
process group is a collection of one or more processes, usually associated with
the same job, that can receive signals from the same terminal. Each process
group has a unique process group ID.

```c
#include <unistd.h>
pid_t getpgrp(void);
// Returns: process group ID of calling process.
```

The Single UNIX Specification defines the `getpgid`.

```c
#include <unistd.h>
pid_t getgpid(pid_t pid);
```

Each process group can have a process group leader. The leader is identified by
its process group ID being equal to its process ID.

It is possible for a process group leader to create a process group, create
processes in the group, and then terminate.

A process joins an existing process group or creates a new process group by
calling `setpgig`

```c
#include <unistd.h>
int setpgid(pid_t pid, pid_t pgid);
```

## Sessions

A session is a collection of one or more process groups. For example, we could
have the arrangement shown below. Here we have three process groups in a single
session.

![Arragement of processes into process groups and sessions](https://i.loli.net/2021/10/10/BVoNLmtpgIa27l6.png)

The processes in a process group are usually placed there by a shell pipeline.

```sh
proc1 | proc2 &
proc3 | proc4 | proc5
```

A process establishes a new session by calling the `setsid` function.

```c
#include <unistd.h>
pid_t setsid(void);
// Returns: process group ID if OK, -1 on error
```

If the process is not a process group leader, this function create a new
session. Three things happen.

+ The process becomes the session leader of this new session.
+ The process becomes the process group leader of a new process group. The new process
  ID is the process ID of the calling process.
+ The process has no controlling terminal. If the process had a controlling terminal
  before calling `setsid`, that association is broken.
  in the foreground process group.
  hang-up signal is sent to the controlling process (the session leader).

## Controlling Terminal

Sessions and process groups have a few other characteristics.

+ A session can have a single *controlling terminal*. This is usually the terminal
  deivce or pseudo terminal device on which we login in.
+ The session leader that establishes the connection to the controlling process
  called the *controlling process*.
+ The process groups within a session can be divided into a single *foreground process group*
and one or more *background process groups*.
+ If a session has a controlling terminal, it has a single foreground process group and
  allow other process groups in the session are background process groups.
+ Whenever we press the terminal's quit key, the quit signal is sent to all processes
  in the foreground process group.
+ If a modem (or network) disconnect is detected by the terminal interface, the
  hang-up signal is sent to the controlling process (the session leader).
  These characteristics are shown below.

These characteristics are shown below.

![Process groups and sessions showing controlling terminal](https://i.loli.net/2021/10/10/kfv6ZoNWDzMdJ3n.png)

## Job Control

Job control allows us to start multiple jobs from a single terminal and to control
which jobs can access the terminal and which jobs are run in the background. Job
control requires these forms of support:

+ A shell that supports job control
+ The terminal driver in the kernel must support job control
+ The kernel must support certain job-control signals

From our perspective, when using job control from a shell, we can start a job in
either the foreground or the background.

When we start a background job, the shell assigns it a job identifier and prints
one or more of the process IDs. And the terminal driver looks for three special
characters, which generate signals to the foreground process group:

+ The interrupt character (Control-C) generates `SIGINT`.
+ The quit character (Control-backslash) generates `SIGQUIT`.
+ The  suspend character (Control-Z) generates `SIGTSTP`.

Since we can have a foreground job and one or more background jobs, which of
these receives the character that we enter at the terminal? Only the foreground
job receives terminal input. The terminal driver detects this and sends a special
signal to the background job: `SIGTTIN`.
