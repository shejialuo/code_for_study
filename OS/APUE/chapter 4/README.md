# Files and Directories

## stat, fstat, fstatat, and lstat Functions

```c
#include <sys/stat.h>
int stat(const char *restrict pathname, struct stat *restrict buf);
int fstat(int fd, struct stat *buf);
int lstat(const char *restrict pathname, struct stat *restrict buf);
int fstatat(int fd, const char *restrict pathname,
            struct stat *restrict buf, int flag);

// All four return: 0 if ok, -1 on error
```

The `buf` argument is a pointer to a structure that we must supply. The
functions fill in the structure. The definition of the structure can
differ among implementations, but it could look like this:

```c
struct stat {
  mode_t          st_mode; /* file type & mode (permissions) */
  ino_t           st_ino; /* i-node number (serial number) */
  dev_t           st_dev; /* device number (file system) */
  dev_t           st_rdev; /* device number for special files */
  nlink_t         st_nlink; /* number of links */
  uid_t           st_uid; /* user ID of owner */
  gid_t           st_gid; /* group ID of owner */
  off_t           st_size; /* size in bytes, for regular files */
  struct timespec st_atim; /* time of last access */
  struct timespec st_mtim; /* time of last modification */
  struct timespec st_ctim; /* time of last file status change */
  blksize_t       st_blksize; /* best I/O block size */
  blkcnt_t        st_blocks; /* number of disk blocks allocated */
};
```

The `timespec` structure type defines time in terms of seconds and nanoseconds.
It includes at least the following fields:

```c
time_t tv_sec;
long   tv_nsec;
```

## File Types

Most files on a UNIX system are either regular files or directories, but there
are additional types of files. The types are

+ Regular file.
+ Directory file.
+ *Block special file*. A type of file providing buffered I/O access in
  fixed-size units to devices such as disk drives.
+ *Character special file*. A type of file providing unbuffered I/O access in
  variable-sized units to devices. All devices on a system are either block
  special files or character special files.
+ *FIFO*. A type of file used for communication between processes.
  It’s sometimes called a named pipe.
+ *Socket*. A type of file used for network communication between processes.
+ *Symbolic link*. A type of file that points to another file.

## Set-User-ID and Set-Group-ID

Every process has six or more IDs associated with it.

+ The real user ID and real group ID identify who we really are. These two fields
  are taken from our entry in the password file when we log in. Normally, these
  values don’t change during a login session, although there are ways for a superuser
  process to change them.
+ The effective user ID, effective group ID, and supplementary group IDs
  determine our file access permissions.
+ The saved set-user-ID and saved set-group-ID contain copies of the effective
  user ID and the effective group ID, when a program is executed.
  Normally, the effective user ID equals the real user ID ,and the effective group
  ID equals the real group ID.

## File Access Permissions

The `st_mode` value also encodes the access permission bits for the file.
The read, write and execute are used in various ways by different functions.
We’ll summarize them here:

+ The first rule is that *whenever* we want to open any type of file by name,
  we must have execute permission in each directory mentioned in the name,
  including the current directory.
+ The read permission for a file determines whether we can open an existing file
  for reading: the `O_RDONLY` and `O_RDWR` flags for the `open` function.
+ The write permission for a file determines whether we can open an existing
  file for writing: the `O_WRONLY` and `O_RDWR` flags for the `open` function.
+ We must have write permission for a file to specify the `O_TRUNC` flag in the
  open function.
+ We cannot create a new file in a directory unless we have write permission and
  execute permission in the directory.
+ To delete an existing file, we need write permission and execute permission in
  the directory containing the file. We do not need read permission or write
  permission for the file itself.
+ Execute permission for a file must be on if we want to execute the file using
  any of the seven `exec` functions. The file also has to be a regular file.

The tests performed by the kernel are as follows:

+ If the effective user ID of the process is 0 (the superuser), access is allowed.
+ If the effective user ID of the process equals the owner ID of the file,
  access is allowed if the appropriate user access permission bit is set.
  Otherwise, permission is denied.
+ If the effective group ID of the process or one of the supplementary group IDs
  of the process equals the group ID of the file, access is allowed if the
  appropriate group access permission bit is set.
+ If the appropriate other access permission bit is set, access is allowed.

## Ownership of New Files and Directories

The user ID of a new file is set to the effective user ID of the process. We can
choose one of the following options to determine the group ID of a new file:

+ The group ID of a new file can be the effective group ID of the process.
+ The group ID of a new file can be the group ID of the directory in which the
  file is being created.

## access and faccessat Functions

As we described earlier, when we open a file, the kernel performs its access
tests based on the effective user and group IDs. Sometimes, a process wants to
test accessibility based on the real user and group IDs.
The `access` and `faccessat` functions base their tests on the real user and
group IDs:

```c
#include <unistd.h>
int access(const char *pathname, int mode);
int faccessat(int fd, const char *pathname, int mode, int flag);
// Both return: 0 if Ok, -1 on error
```

## umask Function

Now that we’ve described the nine permission bits associated with every file, we
can describe the file mode creation mask that is associated with every process.

The `umask` function sets the file mode creation mask for the process and returns
the previous value:

```c
#include <sys/stat.h>
mode_t umask(mode_t cmask);
// Returns: previous file mode creation mask
```

The file mode creation mask is used whenever the process creates a new file or a
new directory.

Most users of UNIX systems deal with their `umask` value. It is usually set once,
on login, by the shell’s start-up file, and never changed. Nevertheless, when
writing programs that create new files, if we want to ensure that specific
access permission bits are enabled, we must modify the `umask` value while the
process is running.

Users can set the `umask` value to control the default permissions on the files
they create. This values is expressed in octal, with one bit representing one
permission to be masked off.

Permissions can be denied by setting the corresponding bits. Some common `umask`
values are 002 to prevent others from writing your files, 022 to prevent group
members and others from writing your files.

## chmod, fchmod, and fchmodat Functions

The `chmod`, `fchmod`, and `fchmodat` functions allow us to change the file
access permissions for an existing file.

```c
#include <sys/stat.h>
int chmod(const char *pathname, mode_t mode);
int fchmod(int fd, mode_t mode);
int fchmodat(int fd, const char *pathname, mode_t mode, int flag);
// All three return: if OK, -1 on error
```

## chown, fchown, fchownat, and lchown Functions

The `chown` functions allow us to change a file’s user ID and group ID, but if
either of the arguments owner or group is -1, the corresponding ID is left unchanged.

```c
#include <unistd.h>
int chown(const char *pathname, uid_t owner, gid_t group);
int fchown(int fd, uid_t owner, gid_t group);
int fchownat(int fd, const char *pathname, uid_t owner, gid_t group,
             int flag);
int lchown(const char *pathname, uid_t owner, gid_t group);
// All four return: 0 if OK, -1 on error.
```

## File Size

The `st_size` member of the `stat` structure contains the size of the file in bytes.
This field is meaningful only for regular files, directories, and symbolic links.

For a regular file, a file size of 0 is allowed. For a symbolic link, the file size
is the number of bytes in the filename.

## link, linkat, unlink, unlinkat, and remove Functions

A file can have mutiple directory entries pointing to its i-node. We can
use either the `link` function or the `linkat` function to create a link
to an existing file.

```c
#include <unistd.h>
int link(const char *existingpath, const char *newpath);
int linkat(int efd, const char *existingpath, int nfd,
           const char *newpath, int flag);
// Both return: 0 if OK, -1 on error
```

If an implementation supports the creation of hard links to directories,
it is restricted to only the superuser. This constraint exists because
such hard links can cause loops in the file system.

To remove an existing directory entry, we call the `unlink` function

```c
#include <unistd.h>
int unlink(const char *pathname);
int unlinkat(int fd, const char *pathname, int flag);
// Both return 0 if OK -1 on error
```

These functions remove the directory entry and decrement the link count
of the file referenced by `pathname`.

We can also unlink a file or a directory with the `remove` function. For
a file, `remove` is identical to `unlink`. For a directory, `remove` is
identical to `rmdir`.

## rename and renameat Functions

A file or a directory is reanmed with either the `rename` or `renameat`
function.

```c
#include <stdio.h>
int rename(const char *oldname, const char *newname);
int rename(int oldfd, const char *oldname,
           int newfd, const char *newname);
// Both return 0 if OK, -1 on error
```

## Symbolic Links

A symbolic link is an indirect pointer to a file, unlike the hard links,
which pointed directly to the i-node of the file. Symbolic links were
introduced to get around the limitations of hard links.

+ Hard links nomrally require that the link and the file reside in the
  same file system.
+ Only the superuser can create a hard link to a directory.

## Create and Reading Symbolic Links

A symbolic link is created with either the `symlink` or `symlinkat`
function.

```c
#include <unistd.h>
int symlink(const char *actualpath, const char *sympath);
int symlink(const char *actualpath, int fd, const char *sympath);
// Both return 0 if OK -1 on error
```

Because the `open` function follows a symbolic link, we need a way to
open the link itself and read the name in the link.

```c
ssize_t readlink(const char *restrict pathname, char *restrict buf,
                 size_t bufsize);
ssize_t readlink(int fd, const char *restrict pathname,
                 char *restrict buf, size_t bufsize);
// Both return number of bytes read if OK, -1 on error
```
