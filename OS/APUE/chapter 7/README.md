# Process Environment

## main Function

A C program starts execution with a function called `main`. The prototype for
the `main` function is:

```c
int main(int agrc, char *argv[]);
```

## Process Termination

There are eight ways for a process to terminate. Normal termination occurs in
five ways:

+ Return from `main`
+ Calling `exit`
+ Calling `_exit` or `_Exit`
+ Return of the last thread from its start routine
+ Calling `pthread_exit` from the last thread

Abnormal termination occurs in three ways:

+ Calling `abort`
+ Receipt of a signal
+ Response of the last thread to a cancellation request

### Exit Functions

Three functions terminate a program normally: `_exit` and `_Exit`, which return
to the kernel immediately, and `exit`, which performs certain cleanup processing
and then returns to the kernel.

```c
#include <stdlib.h>
void exit(int status);
void _Exit(int status);
#include <unistd.h>
void _exit(int status);
```

Historically, the `exit` function has always performed a clean shutdown of the
standard I/O library: the `fclose` function is called for all open streams.

All three exit functions expect a single integer argument, which we call the
*exit status*.

### atexit Function

With ISO C, a process can register at least 32 functions that are automatically
called by `exit`. These are called *exit handlers* and are registered by calling
the `atexit` function.

```c
#include <stdlib.h>
int atexit(void (*func)(void));
```

The below figure summarizes how a C program is started and the various ways it
can terminate:

![How a C program is started and how it terminates](https://i.loli.net/2021/05/16/yNEuSBRHWjgG4tq.png)

## Command-Line Arguments

When a program is executed, the process that does the `exec` can pass command-line
arguments to the new program.

```c
#include <stdio.h>
int main(int argc, char *argv[]) {
  int i;
  for(i = 0; i < argc; ++i) {
    printf("argv[%d]: %s\n", i, argv[i]);
  }
}
```

## Environment List

Each program is also passed an *environment list*:

```c
extern char **environ;
```

## Memory Layout of a C Program

Historically, a C program has been composed of the following pieces:

+ Text segment
+ Initialized data segment also called data segment
+ Uninitialized data segment, often called the "bss" segment
+ Stack
+ Heap

![Typical memory arrangement](https://i.loli.net/2021/05/16/hBfMzWJZwX2R6GK.png)

## Memory Allocation

ISO C specifies three functions for memory allocation:

+ `malloc`, which allocates a specified number of bytes of memory. The initial
  value of the memory is indeterminate.
+ `calloc`, which allocates space for a specified number of objects of a specified
  size. The space is initialized to all 0 bits.
+ `realloc`, which increases or decreases the size of a previously allocated area.
  When the size increases, it may involve the previously allocated area somewhere
  else, to provide the additional room at the end. Also, when the size increases,
  the initial value of the space between the old contents and the end of the new
  area is indeterminate.

```c
#include <stdlib.h>
void *malloc(size_t size);
void *calloc(size_t nobj, size_t size);
void *realloc(void *ptr, size_t newsize);
// All three return: non-null pointer if OK, NULL on error
void free(void *ptr);
```

## Environment Variables

ISO C defines a function that we can use to fetch values from the environment,
but this standard says that the contents of the environment are implementation defined.

```c
#include <stdlib.h>
char *getenv(const char *name);
// Returns: pointer to value associated with name, NULL if not found
```

We should always use `getenv` to fetch a specific value from the environment,
instead of accessing `environ` directly.

In addition to fetching the value of an environment variable, sometimes we may
want to set an environment variable:

```c
#include <stdlib.h>
char *getenv(const char *name);
// Returns: pointer to value associated with name, NULL if not found
```

In addition to fetching the value of an environment variable, sometimes we may
want to set an environment variable.

```c
#include <stdlib.h>
int putenv(char *str);
// Returns: 0 if OK, nonzero on error

int setenv(const char *name, const char*value, int rewrite);
int unsetenv(const char *name);
// Both return: 0 if OK, -1 on error
```

It is interesting to examine how these functions must operate when modifying the
environment list:

+ If we're modifying an existing name:
  + If the size of the new value is less than or equal to the size of the existing
    value, we can just copy the new string over the old string.
  + If the size of the new value is larger than the old one, however, we must
    `malloc` to obtain room for the new string, copy the new string to this area,
    and then replace the old pointer in the environment list for name with the pointer
    to this allocated area.
+ If we're adding a new name, it's more complicated. First, we have to call
  `malloc` to allocate room for the `name=value` string and copy the string to this
  area.
  + If it's the first time we've added a new name, we have to call `malloc` to
    obtain room for a new list of pointers. We copy the old environment list to the
    new area and store a pointer to the `name=value` string at the end of the list
    of pointers. We also store a null pointer to this new list of pointers. Finally,
    we set `environ` to point to this new list of pointers.
  + If this isn't the first time we've added new strings to the environment list,
    that we know we've already allocated room for the list on the heap, so we just
    call `realloc` to allocate room for one more pointer.

## setjmp and longjmp Functions

In C, we can't `goto` a label that's in another function. Instead, we must use
the `setjmp` and `longjmp` functions to perform this type of branching. As we'll
see, these two functions are useful for handling error conditions that occur in
a deeply nested function call.

Consider the following example:

```c
#define TOK_ADD 5
#define MAXLINE 4096
void do_line(char *);
void cmd_add();
int get_token();

int main() {
  char line[MAXLINE];
  while(fgets(line, MAXLINE, stdin) != NULL)
    do_line(line);
  exit(0);
}

char *tok_ptr;

void do_line(char *ptr) {
  int cmd;
  tok_ptr = ptr;
  while((cmd = get_token() > 0)) {
    switch(cmd) {
      case TOK_ADD:
        cmd_add();
        break;
    }
  }
}

void cmd_add() {
  int token;
  token = get_token();
}

int get_token() {
  /* fetch next token from line pointed to by tok_ptr */
}
```

In this code snippet, if the `cmd_add` function encounters an error, it might
want to print an error message, ignore the rest of the input line, and return to
`main` function to read the next input line.

The solution to this problem is to use a nonlocal `goto`:

```c
#include <setjmp.h>
int setjmp(jmp_buf env);
// Returns: 0 if called directly, nonzero if returnning from a call to longjmp
void longjmp(jmp_buf env, int val);
```

Let's return to the example:

```c
#include <setjmp.h>
jmp_buf jmpbuffer;

int main() {
  char line[MAXLINE];
  while(fgets(line, MAXLINE, stdin) != NULL)
    do_line(line);
  exit(0);
}
...
void cmd_add() {
  int token;
  token = get_token();
  if(token < 0)
    longjmp(jmpbuffer, 1);
}
```

## getrlimit and setrlimit Functions

Every process has a set of `getrlimit` and `setrlimit` Functions

Every process has a set of resource limits, some of which can be queried and
changed by the `getrlimit` and `setrlimit` functions.

```c
#include <sys/resource.h>
int getrlimit(int resource, struct rlimit *rlptr);
int setrlimit(int resource, const struct rlimit *rlptr);
// Both return: 0 if OK, -1 on error
```

Each call to these two functions specifies a single *resource* and a pointer
to the following structure:

```c
struct rlimit {
  rlim_t rlim_cur; // soft limit: current limit.
  rlim_t rlim_max; // hard limit: maximum value for rlim_cur.
}
```

Three rules govern the changing of the resource limits:

1. A process can change its soft limit to a value less than or equal to its hard limit.
2. A process can lower its hard limit to a value greater than or equal to its soft limit.
This lowering of the hard limit is irreversible for normal users.
3. Only a superuser process can raise a hard limit.
