# Standard I/O Library

## Streams and FILE Objects

With the standard I/O library, the discussion centers on *streams*. When we
open or create a file with the standard I/O library, we say that we have
associated a stream with the file.

With the ASCII character set, a single character is represented by a single byte.
With international character sets, a character can be represented by more than
one byte. Standard I/O file streams can be used with both single-byte and
multibyte ("wide") character sets.

A stream's orientation determines whether the characters that are read and
written are single byte or multibyte. Initially, when a stream is created,
it has no orientation. If a multibyte I/O function (see `<wchar.h>`) is used on
a stream without orientation, the stream's orientation is set to wide oriented.
If a byte I/O function is used on a stream without orientation, the stream's
orientation is set to byte oriented.

Only two functions can change the orientation once set.
The `freopen` function will clear a stream's orientation; the `fwide` function
can be used to set a stream's orientation.

```c
#include <stdio.h>
#include <wchar.h>
int fwide(FILE *fp, int mode);
/* Returns: positive if stream is wide oriented,
            negative if stream is byte oriented,
            or 0 if stream has no orientation */
```

The `fwide` function performs different tasks, depending on the value of the
`mode` argument.

+ If the `mode` argument is negative, `fwide` will try to make the specified
  stream byte oriented.
+ If the `mode` argument is positive, `fwide` will try to make the specified
  stream wide oriented.
+ If the `mode` argument is zero, `fwide` will not try to set the orientation,
  but will still return a value identifying the stream's orientation.

Note that `fwide` will not change the orientation of a stream that is already
oriented. Also note that there is no error return.

## Standard Input, Standard Output, and Standard Error

Three streams are predefined and automatically available to a process: standard
input, standard output, and standard error.

These three standard I/O streams are referenced through the predefined file
pointers `stdin`, `stdout`, and `stderr`. The file pointers are defined in
the `<stdio.h>` header.

## Buffering

The goal of the buffering provided by the standard I/O library is to use the
minimum number of `read` and `write` calls. Also, this library tries to do its
buffering automatically for each I/O stream, obviating the need for the
application to worry about it.

There are three types of buffering:

+ Fully buffered. In this case, actual I/O takes place when the standard I/O
  buffer is filled. The buffer used is usually obtained by one of the standard
  I/O functions calling `malloc` the first time I/O is performed on a stream.
  The term *flush* describes the writing of a standard I/O buffer.
+ Line buffered. In this case, the standard I/O library performs I/O when a
  newline character is encountered on input or output.
+ Unbuffered. The standard I/O library does not buffer the characters.

ISO C requires the following buffering characteristics:

+ Standard input and standard output are fully buffered, if and only they do
  not refer to an interactive device.
+ Standard error is never fully buffered.

Most implementations default to the following types of buffering:

+ Standard error is always unbuffered.
+ All other streams are line buffered if they refer to a terminal device;
  otherwise, they are fully buffered.

We can change the buffering by calling either the `setbuf` or `setvbuf` function.

```c
#include <stdio.h>
void setbuf(FILE *restrict fp, char *restrict buf);
int setvbuf(FILE *restrict fp, char *restrict buf, int mode,
            size_t size);
// Returns: 0 if OK, nonzero on error
```

At any time, we can force a stream to be flushed.

```c
#include <stdio.h>
int fflush(FILE *fp);
// Returns: 0 if OK, EOF on error
```

The `fflush` function causes any unwritten data for the stream to be passed to
the kernel. As a special case, if `fp` is `NULL`, `fflush` causes all output
streams to be flushed.

## Opening a Stream

The `fopen`, `freopen`, and `fdopen` functions open a standard I/O stream.

```c
#include <stdio.h>
FILE *fopen(const char *restrict pathname, const char *restrict type);
FILE *freopen(const char *restrict pathname, const char *restrict type,
              FILE *restrict fp);
FILE *fdopen(int fd, const char *type);
// All three return: file pointer if OK, NULL on error
```

The differences in these three functions are as follows:

+ The `fopen` function opens a specified file.
+ The `freopen` function opens a specified file on a specified stream, closing
  the stream first if it is already open. If the stream previously had an
  orientation, `freopen` clears it.
+ The `fdopen` function takes an existing file descriptor. This function is
  often used with descriptors that are returned by the functions that create
  pipes and network communication channels.

By default, the stream that is opened is fully buffered, unless it refers to a
terminal device, in which case it is line buffered.

## Reading and Writing a Stream

Once we open a stream, we can choose from among three types of unformatted I/O:

+ Character-at-a-time I/O.
+ Line-at-a-time I/O.
+ Direct I/O. This type of I/O is supported by the `fread` and `fwrite`
  functions. For each I/O operation, we read or write some number of objects,
  where each object is of a specified size.

### Input Functions

Three functions allow us to raed one character at a time:

```c
#include <stdio.h>
int getc(File *fp);
int fgetc(File *fp);
int getchar();
// All three return: next character if OK, EOF on end of file or error
```

These three functions return the next character as an `unsigned char` converted
to an `int`.

Note that these functions return the same value whether an error occurs or the
end of file is reached. To distinguish between the two, we must call either
`ferror` or `feof`.

```c
#include <stdio.h>
int ferror(FILE *fp);
int feof(FILE *fp);
// Both return: nonzero (true) if condition is true, 0 (false) otherwise
void clearerr(FILE *fp);
```

In most implementations, two flags are maintained for each stream in the `FILE` object:

+ An error flag
+ An end-of-file flag

Both flags are cleared by calling `clearerr`.

### Output Functions

```c
#include <stdio.h>
int putc(int c, FILE *fp);
int fputc(int c, FILE *fp);
int putchar(int c);
// All three return: c if OK, EOF on error
```

## Line-at-a-Time I/O

Line-at-a-time input is provided by the two functions, `fgets` and `gets`.

```c
#include <stdio.h>
char *fgets(char *restrict buf, int n, FILE *restrict fp);
char *gets(char *buf);
```

Both specify the address of the buffer to read the line into. The `gets` function
reads from standard input, whereas `fgets` reads from the specified stream.
The `gets` function should never be used. The problem is that it doesn't
allow the caller to specify the buffer size.

Line-at-a-time output is provided by `fputs` and `puts`

```c
#include <stdio.h>
int fputs(const char *restrict str, FILE *restrict fp);
int puts(const char *str);
```

## Binary I/O

If we're doing binary I/O, we often would like to read or write an entire structure
at a time. To do this using `getc` or `putc`, we have to loop through the entire
structure. We can't use the line-at-a-time functions, since `fputs` stops
writting when it hits a null byte. Therefore, the following two functions are
provided for binary I/O.

```c
#include <stdio.h>
size_t fread(void *restrict ptr, size_t size, size_t nobj,
             FILE *restrict fp);
size_t fwrite(const void *restrict ptr, size_t size, size_t nobj,
              FILE *restrict fp);
// Both return: number of objects read or written
```

These functions have two common uses:

Read or write a binary array. For example, to write elements 2 through 5 of a
floating-point array, we could write:

```c
float data[10];
fwrite(&data[2], sizeof(float), 4 ,fp);
```

Read or write a structure. For example, we could write

```c
struct {
  short count;
  long total;
  char name[NAMESIZE];
} item;

fwrite(&item, sizeof(item), 1, fp)
```

A fundamental problem with binary I/O is that it can be used to read only data
that has been written on the same system:

+ The offset of a member within a structure can differ between compilers and
  systems because of different alignment requirements.
+ The binary formats used to store multibyte integers and floating-point values
  differ among machine architectures.

## Positioning a Stream

There are three ways to position a standard I/O stream:

+ The two functions `ftell` and `fseek`.
+ The two functions `ftello` and `fseeko`.
+ The two functions `fgetpos` and `fsetpos`.

```c
#include <stdio.h>
long ftell(FILE *fp);
// Returns: current file position indicator if OK, -1L on error
int fseek(FILE *fp, long offset, int whence);
// Returns: 0 if OK, -1 on error
void rewind(FILE *fp);
```

```c
off_t ftello(FILE *fp);
int fseeko(FILE *fp, off_t offset, int whence);
```

```c
#include <stdio.h>
int fgetpos(FILE *restrict fp, fpos_t *restrict pos);
int fsetpos(FILE *fp, const fpos_t *pos);
// Both return: 0 if OK, nonzero on error
```

The `fgetpos` function stores the current value of the file’s position indicator
in the object pointed to by `pos`. This value can be used in a later call to
`fsetpos` to reposition the stream to that location.

## Implementation Details

We can obtain the descriptor for a stream by calling `fileno`:

```c
#include <stdio.h>
int fileno(FILE *fp);
// Returns: the file descriptor associated with the stream
```

## Temporary Files

The ISO C standard defines two functions that are provided by the standard I/O
library to assist in creating temporary files.

```c
#include<stdio.h>
char *tmpnam(char *ptr);
// Returns: pointer to unique pathname
FILE *tmpfile(void);
// Returns: file pointer if OK, NULL on error
```

The `tmpnam` function generates a string that is a valid pathname and that does
not match the name of any existing file. The `tmpfile` function creates a temporary
binary file that is automatically removed when it is closed or on program termination.

The Single UNIX Specification defines two additional functions as part of the XSI
option for dealing with temporary files: `mkdtemp` and `mkstemp`:

```c
#include<stdlib.h>
char *mkdtemp(char *template);
// Returns: pointer to directory name if OK, NULL on error.
int mkstemp(char *template);
// Returns: file descriptor if OK, -1 on error.
```
