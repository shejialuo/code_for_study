# 6. Using Linux Environment Variables

## 6.1 Exploring Environment Variables.

There are two environment variable types in the bash shell:

+ Global variables
+ Local variables

### 6.1.1 Looking at global environment variables

Global environment variables are visible from the shell session and
from any spawned child subshells.

The Linux system sets several global environment variables when you
start your bash session.

To view global environment variables, use the `env` or the `printenv` command.

To display an individual environment variable's value, you can use
the `printenv` command, but not the `env` command.

```sh
printenv HOME
```

### 6.1.2 Looking at local environment variables

Trying to see the local variables list is a little tricky at the CLI.
Unfortunately, there isn't a command that displays only these variables.
The `set` command displays all variables defined for a specific process,
including both local and global environment variables and user-defined
variables.

## 6.2 Setting User-Defined Variables

### 6.2.1 Setting local user-defined variables

```sh
my_variable1=Hello
my_variable2="Hello World"
```

### 6.2.2 Setting global environment variables

The method used to create a global environment variables is to
first create a local variable and then export it to the global environment.

```sh
my_variable="I am global now"
export my_variable
```

## 6.3 Removing Environment Variables

You can remove an existing environment variable with the `unset` command.

```sh
my_variable="I am global now"
echo $my_variable
unset my_variable
echo $my_variable
```

## 6.4 Uncovering Default Shell Environment Variables

There are too many variables, I choose only some:

+ `IFS`: A list of characters that separate fields used by the shell
to split text strings.
+ `PATH`: A colon-separated list of directories where the shell
looks for commands.
+ `DIRSTACK`: An array variable that contains the current contents
of the directory stack.
+ `FUNCNAME`: The name of the currently executing shell function.
+ `FUNCNEST`: Sets the maximum allowed function nesting level.
+ `HOSTNAME`: The name of the current host.
+ `LANG`: The locale category for the shell.
+ `LC_ALL`: Overrides the `LANG` variable, defining a locale category.
+ `LC_CTYPE`: Determines the interpretation of characters used in
filename expansion and pattern matching.
+ `PWD`: The current working directory.
+ `SHELL`: The full pathname to the shell.
+ `UID`: The numeric real user ID of the current user.

## 6.5 Setting the PATH Environment Variable

The `PATH` environment variable defines the directories it
searches looking for commands and programs.

You can add new search directories to existing `PATH` environment variable
without having to rebuild it from scratch.

```sh
PATH="$HOME/.local/bin:$PATH"
```

## 6.6 Locating System Environment Variables

When you start a bash shell by logging in to the Linux system,
by default bash checks several files for commands. These files
are called *startup files* or *environment files*. You can start a
bash shell in three ways:

+ As a default login shell at login time
+ As an interactive shell that is started by spawning a subshell.
+ As a non-interactive shell to run a script

When you log in to the Linux system, the bash shell starts as a
login shell. The login shell typically looks for five different
startup files to process commands from:

+ `/etc/profile`
+ `$HOME/.profile`
+ `$HOME/.bashrc`
+ `$HOME/.bash_profile`
+ `$HOME/.bash_login`

## 6.7 Learning about Variable Arrays

To set multiple values for an environment variable, just list them
in parentheses, with values separated by spaces:

```sh
mytest=(one two three four five)
echo $mytest # one
echo ${mytest[2]} #three
```

To display an entire array variable, you use the asterisk wildcard
character as the index value:

```sh
echo ${mytest[*]} # one two three four five
```
