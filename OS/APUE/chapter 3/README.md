# File I/O

## File Descriptors

To the kernel, all open files are referred to by file descriptors. A file
descriptor is a non-negative integer. When we open an existing file or create a
new file, the kernel returns a file descriptor to the process.

By convention, UNIX system shells associate file descriptor 0 with the standard
input of a process, file descriptor 1 with the standard output, and file
descriptor 2 with the standard error.

## open and openat Functions

A file is opened or created by calling either the `open` function or the
`openat` function:

```c
#include <fcntl.h>
int open(const char* path, int oflag, .../* mode_t mode */);
int openat(int fd, const char* path, int oflag, .../* mode_t mode */);
// Both return: file descriptor if OK, -1 on error
```

## creat Function

A new file can also be created by calling the `creat` function:

```c
#include <fcntl.h>
int creat(const char* path, mode_t mode);
// Returns: file descriptor opened for write-only if OK, -1 on error
```

## close Function

An open file is closed by calling the `close` function:

```c
#include <unistd.h>
int close(int fd);
// Returns: 0 if Ok, -1 on error
```

## lseek Function

Every open file has an associated "current file offset", normally a non-negative
integer that measures the number of bytes from the beginning of the file.
By default, this offset is initialized to 0 when a file is opened, unless
the `O_APPEND` option is specified.

An open file's offset can be set explicitly by calling `lseek`:

```c
#include <unistd.h>
off_t lseek(int fd, off_t offset, int whence);
// Returns: new file offset if OK, -1 on error
```

The interpretation of the `offset` depends on the value of the `whence` argument:

+ If `whence` is `SEEK_SET`, the file's offset is set to `offset` bytes from the
  beginning of the file.
+ If `whence` is `SEEK_CUR`, the file's offset is set to its current value plus
  the `offset`. The `offset` can be positive or negative.
+ If `whence` is `SEEK_END`, the file's offset is set to the size of the file
  plus the `offset`. The `offset` can be positive or negative.

Because a successful call to `lseek` returns the new file offset, we can seek
zero bytes from the current position to determine the current offset:

```c
off_t currpos;
corrpos = lseek(fd, 0, SEEK_CUR);
```

If the file descriptor refers to a pipe, FIFO, or socket, `lseek` sets
`errno` to `ESPIPE` and returns -1. For example:

```c++
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
using namespace std;

int main() {
  if(lseek(STDIN_FILENO, 0, SEEK_CUR) == -1)
    cout << "cannot seek\n";
  else {
    cout << "seek OK\n";
  }
  return 0;
}
```

The file's offset can be greater than the file's current size, in which case the
next `write` to the file will extend the file. This is referred to as creating a
hole in a file and is allowed. Any bytes in a file that have not been written are
read back as 0.

## read Function

Data is read from an open file with the `read` function:

```c
#include <unistd.h>
ssize_t read(int fd, void *buf, size_t nbytes)
// Returns: number of bytes read, 0 if the end of file, -1 on error
```

## write Function

Data is written to an open file with the `write` function:

```c
#include <unistd.h>
ssize_t write(int fd, const void *buf, size_t nbytes);
```

## File Sharing

The UNIX System supports the sharing of open files among different processes.
The kernel uses three data structure to represent an open file, and the
relationships among them determine the effect one process has on another with
regard to file sharing.

+ Every process has an entry in the process table. Within each process table
  entry is a table of open file descriptors. Associated with each file descriptor
  are:
  + The file descriptor flags
  + A pointer to a file table entry
+ The kernel maintains a file table for all open files. Each file table entry contains
  + The file status flags for the file
  + The current file offset
  + A pointer to the v-node table entry for the file
+ Each open file has a v-node structure that contains information about the type
  of file and pointers to functions that operate on the file. For most files, the
  v-node also contains the i-node for the file. This information is read from disk
  when the file is opened, so that all the pertinent information about the file is
  readily available.

If two independent processes have the same file open,
we could have the arrangement shown below.

![Two independent processes with the same file open](https://i.loli.net/2021/05/16/E12oTYDnJLFUOgI.png)

Given these data structures, we now need to be more specific about what happens
with certain operations:

+ After each `write` is complete, the current file offset in the file table entry
  is incremented by the number of bytes written. If this causes the current file
  offset to exceed the current file size, the current file size in the i-node table
  entry is set to the current file offset.
+ If a file is opened with the `O_APPEND` flag, a corresponding flag is set in
  the file status flags of the file table entry. Each time a `write` is performed
  for a file with this append flag set, the current file offset in the file table
  entry is set the current file size from the i-node table entry.
+ The `lseek` function modifies only the current file offset in the file table
  entry. No I/O takes place.

It's possible for more than one file descriptor entry to point to the same file
table entry. This also happens after a `fork` when the parent and child share
the same file table entry for each open descriptor.

## Atomic Operations

The Single Unix Specification includes two functions that allow applications to
seek and perform I/O atomically: `pread` and `pwrite`:

```c
#include <unistd.h>
ssize_t pread(int fd, void *buf, size_t nbytes, off_t offset);
ssize_t pwrite(int fd, const void *buf, size_t nbytes, off_t offset);
```

## dup and dup2 Functions

An existing file descriptor is duplicated by either of the following functions:

```c
#include <unistd.h>
int dup(int fd);
int dup2(int fd, int fd2);
// Both return: new file descriptor if OK, -1 on error
```

The new file descriptor returned by the `dup` is guaranteed to be the
lowest-numbered available file descriptor. With `dup2`, we specify the value of
the new descriptor with the `fd2` argument. If `fd2` is already open, it is
first closed. If `fd` equals `fd2`, then `dup2` returns `fd2` without closing it.

The new file descriptor that is returned as the value of the functions shares
the same file table entry as the `fd` argument .

![Kernel data structures after dup](https://i.loli.net/2021/05/16/aNprDzfwh7GqPMe.png)

## sync, fsync, and fdatasync Functions

Traditional implementations of the UNIX System have a buffer cache or page cache
in the kernel through which most disk I/O passes. When we write data to a file,
the data is normally copied by the kernel into one of its buffers and queued for
writing to disk at some later time. This is called **delayed write**.

The kernel eventually writes all the delayed-write blocks to disk, normally when
it needs to reuse the buffer for some other disk block.
To ensure the consistency of the file system on disk with the contents of the
buffer cache, the `sync`, `fsync`, and `fdatasync` functions are provided.

```c
#include <unistd.h>
int fsync(int fd);
int fdatasync(int fd);
void sync(void);
```

The `sync` function simply queues all the modified block buffers for writing and
returns; it does not wait for the disk writes to take place.

The function `fsync` refers only to a single file, specified by the file
descriptor `fd`, and waits for the disk writes to complete before returning.

The `fdatasync` function is similar to `fsync`, but it affects only the
data portions of a file.

## fcntl Function

The `fcntl` function can change the properties of a file that is already open.

```c
#include <fcntl.h>
int fcntl(int fd, int cmd, ... /* int arg */);
// Returns: depends on cmd if OK, -1 on error
```

The `fcntl` function is used for five different purposes.

+ Duplicate an existing descriptor. (`cmd = F_DUPFD` or `F_DUPFD_CLOEXEC`)
+ Get/set file descriptor flags. (`cmd = F_GETFD` or `F_SETFD`)
+ Get/set file status flags. (`cmd = F_GETFL` or `F_SETFL`)
+ Get/set asynchronous I/O ownership.
+ Get/set record locks.

## ioctl Function

The `ioctl` function has always been the catchall for I/O operations. Anything
that couldn't be expressed using one of the other functions in this chapter
usually ended up being specified with an `ioctl`.

```c
#include <unistd.h>
#include <sys/ioctl.h>
int ioctl(int fd, int request, ...);
// Returns: -1 on error, something else if OK.
```

## /dev/fd

Newer systems provide a directory named `/dev/fd` whose entries are files named
0, 1, 2 and so on. Opening the file `dev/fd/n` is equivalent to duplicating
descriptor n, assuming that descriptor n is open.

In the function call: `fd = open("/dev/fd/0", mode);`, most systems ignore the
specified `mode`, whereas others require that it be a subset of the mode used
when the referenced file was originally opened.
