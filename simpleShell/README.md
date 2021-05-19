# Simple Shell

This is a simple shell for studying. Actually, It is the
project work of Operating Systems Three Easy Pieces
[UNIX Shell](https://github.com/remzi-arpacidusseau/ostep-projects),
you can find the detail here.

## Functionality

There are three built-in shell command:

+ `cd`: Like the normal shell does. You can just use one parameter.
+ `exit`: Simply exit the shell. You cannot use any parameters.
+ `path`: It is important, it provides the path for shell to search path.
          And it accepts zero or more parameters, do remember that if the
          user sets path to be empty, then the shell should not be able to run any programs (except built-in commands).

There are two mode:

+ Interactive
+ Batch

However, there are some I don't finish:

+ Parallel Commands:
  
  ```shell
  wish> cmd1 & cmd2 args1 args2 & cmd3 args1
  ```

+ Redirection

  ```shell
  ls -la /tmp > output
  ```

## Build

It's easy, just use `g++`:

```shell
g++ -g simpleShell.cpp -o wish
```
