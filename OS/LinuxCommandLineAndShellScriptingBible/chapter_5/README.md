# 5. Understanding the Shell

A shell is not just a CLI. It is a complicated interactive running
program.

## 5.1 Exploring Shell Types

The shell program that the system starts depends on your user ID
configuration. In the `/etc/passwd` file, the user ID has its default
shell program list in field `#7`. of its record.

The *default interactive shell* starts whenever a user logs into
a virtual console terminal or starts a terminal emulator in the GUI.
However, another default shell, `/bin/sh`, is the *default system shell*.
The default system shell is used for system shell scripts, such as
those needed at startup.

## 5.2 Exploring Parent and Child Shell Relationships

The default interactive shell started when a user logs into a virtual
console terminal or starts a terminal emulator in the GUI is a *parent shell*.

When the `bash` command is entered at the CLI prompt, a new shell program
is created. This is a *child shell*.

When a child shell process is spawned, only some of the parent's environment
is copied to the child shell environment. This can cause problems
with items such as variables.

## 5.3 Understanding Shell Built-In Commands

### 5.3.1 Looking at external commands

An *external command* is a program that exists outside of the
bash shell. THey are not built into the shell program.

The `ps` command is an external command. You can find its filename
by both the `which` and the `type` commands:

```sh
which ps
type -a ps
```

Because it is an external command, when the `ps` command executes,
a child process is created.

### 5.3.2 Looking at built-in commands

*Built-in commands* are different in that they do not need a child
process to execute. They were complied into the shell and thus are
part of the shell's toolkit.

Both the `cd` and `exit` are built into the bash shell.
