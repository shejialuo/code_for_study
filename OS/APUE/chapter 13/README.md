# Daemon Processes

## Introduction

Daemons are processes that live for a long time. In this chapter, we look at
the process structure of daemons and explore how to write a daemon. Since a
daemon does not have a controlling terminal, we need to see how a daemon can
report error conditions when something goes wrong.

## Daemon Characteristics

In the `ps` command, kernel daemons appear with their names in square brackets. Linux
uses a special kernel process, `kthreadd`, to create other kernel processes, so
`kthreadd` appears as the parent of the other kernel daemons. Each kernel component
that needs to perform work in a process context, but that isn't invoked from
the context of a user-level process, will usually have its own kernel daemon.

Note that most of the daemons run with root privileges. None of the daemons has
a controlling terminal: the terminal name is set to a question mark. The kernel daemons
are started without a controlling terminal. The lack of a controlling terminal
in the user-level daemons is probably the result of the daemons having called `setsid`.
Most of the user-level daemons are process group leaders and session leaders, and
are the only processes in their process group and session. Finally, note that
the parent of the user-level daemons is the `init` process.

## Coding rules

Some basic rules to coding a daemon prevent unwanted interactions from happening.

+ Call `umask` to set the file mode creation mask to a known value, usually 0.
+ Call `fork` and have the parent `exit`. This does several things. First,
  if the daemon was started as a simple shell command, having the parent terminate
  makes shell think that the command is done. Second, the child inherits the process
  group ID
  of the parent but gets a new process ID, so we're guaranteed that the child
  is not a process group leader.
+ Call `setsid` to create a new session.
+ Changing the current working directory to the root directory.
+ Unneeded file descriptors should be closed.

## Error Logging

One problem a daemon has is how to handle error messages. It can't simply write
to standard error,
since it shouldn't have a controlling terminal. We don't want all the daemons writing
to the console device. We also don't want each daemon writing its own error messages
into a separate file.
A central daemon error-logging facility is required.

![The syslog facility](https://i.loli.net/2021/10/12/orhi3ABaTgQy2nd.png)

```c
#include <syslog.h>
void openlog(const char *ident, int option, int facility);
void syslog(int priority, const char *foarmt, ...);
void closelog(void);
int setlogmask(int maskpri);
```
