# Chapter 5 Practical matters: strings, loops, and history

## 5.1 Strings

Strings are tremendously important in gnuplot, because they serve three different purposes:

+ *Strings as descriptions*
+ *Strings as data structures*
+ *Strings as marcos and functions*

### 5.1.1 Quotes

You can assign a string constant to a gnuplot variable. String constants must be
enclosed in quotes (either single or double).

The difference is that escaped control characters are interpreted as control characters
within *double*-quoted strings but are treated literally inside *single*-quoted strings.

A double-quoted string can contain single quotes; to obtain a double quote inside
a double-quoted string, it must be scaped with a preceding backslash.

```gnuplot
a = 'This is a string.'
b = "First Line\nSecond Line"

c = "Double quote\" escaped."
d = 'Single quote'' escaped.'
```

### 5.1.2 String operations

Strings can be assigned to variables just as numbers can. Strings are converted to
numbers silently, if possible. But only integers are promoted to strings:

```gnuplot
x = '3.14'
y = 2 + x

a = 4
b = 'foo' . a

z = 0.0 + x
c = '' . z
```

Three operators and a handful of functions act on strings. The first operator
is the concatenation or *dot* operator:

```gnuplot
a = 'baz'
b = "foo" . 'bar'
c = b . a
```

The other two operators are comparison operators for use in conditionals: `eq` and `ne`.

Strings can be indexed like arrays, using a syntax similar to the syntax
used to indicate plot ranges.

```gnuplot
a =  "Gnuplot"
b = a[2:4]
c = a[4:]
```

The first character in the string has index 1. Also notice that *both* limits are inclusive.
If you leave either the beginning or the end of the substring empty,
it will default to the beginning or the end of the entire string.

## 5.2 String expressions and string macros

The string-handling capabilities just introduced make it possible
to construct command lines programmatically. Specifically,

+ Wherever a command expects a string, a string expression can be substituted.
+ The `eval` command takes a fixed string and executes it as if
its contents were entered at the command line.
+ By prefixing a variable containing a string with the `@` character,
you can create macros that can be used even in places where gnuplot doesn't
expect to find a string expression.

### 5.2.1 String expressions in commands

Anywhere a command expects a string, you can substitute a *string expression*.

```gnuplot
file = "data.txt"
desc = "My Data"

plot file title desc
```

You can also use functions that return a string. Here, a function
provides the file extension for you:

```gnuplot
f(x) = x . '.txt'
plot f('data')
```

### 5.2.2 Executing a string with eval

The `eval` command takes a string and executes it as if its contents had been
issued at the command prompt.

### 5.2.3 String macros inside commands

When you want to replace part of a command line where gnuplot doesn't expect a
string, you can't use a string variable directly. Instead, you must resort to
a string *macro*. String macros let you insert the contents of string
variables at arbitrary positions in a gnuplot command.

```gnuplot
cols = "using 1:3"
style = "with lines"

plot "data" @cols @style
```

Macros aren't evaluated within quotes (either single or double), but you can usually
achieve the desired effect through simple string manipulations.

```gnuplot
tool = "(made with gnuplot)"

set title "My plot @tool"
set title "My plot " . tool
```

## 5.3 Generating textual output

Gnuplot has two facilities for generating text: the `print` command and the
`set table` option. Although they may seem similar, they're intended for
different uses:

+ The `print` command evaluates expressions and displays their results as text.
+ The `table` facility generates a textual description of the output of the `plot`
or `splot` command.

### 5.3.1 The print and set print commands

The `print` command evaluates one or more expressions (separated by commas) and prints
them to the currently active printing channel:

```gnuplot
print sin(1.5*pi)
print "The value of pi is: ", pi
```

This example demonstrates the two typical uses for the `print` command:

+ Evaluating expressions (effectively, using gnuplot as a calculator)
+ Inspecting the current value of variables and heredocs

You can change the device to which `print` sends its output by using the `set print` option:

```gnuplot
set print
set print "-"
set print "{str: filename}" [append]
set print $heredoc [append]
```

By default, `print` sends its output to standard error. Each invocation of `set print`
creates a new file, unless you specify the additional keyword `append`.

### 5.3.2 The set table command and the with table style

The `set table` facility gives you access to the data that makes up
a graph *as text*:

```gnuplot
set table ["{str:filename}" | $heredoc]
```

### 5.3.3 Reading and writing heredocs

No explicit commands exist to save the contents of a heredoc to a file or to
populate a heredoc from a general data file, but both tasks acn be accomplished through
creative use of the techniques just discussed.

Remember that the contents of a heredoc aren't included in the information persisted with `save`.

```gnuplot
set print "data"
print $d
unset print
```

```gnuplot
set table $d
plot "data" with table
unset table
```

## 5.4 Simplifying work with inline loops

### 5.4.1 Loops over numbers

Let's assume that you wish to plot the second, third and forth columns of a file
against the first:

```gnuplot
plot for [j=2:4] "data" u 1:j
```

The loop condition is specified in square brackets:

```gnuplot
for [variable = start : end: increment]
```

### 5.4.2 Loops over strings

For example:

```gnuplot
plot "newyork" u 1:2, "chicago" u 1:2, "sanfranciso" u 1:2
```

It is tedious, so we can loop over strings to simplify:

```gnuplot
plot for [name in "newyork chicago sanfranciso"] name u 1:2
```

The loop condition in this case is:

```gnuplot
for [variable in tokenstring]
```

## 5.5 Gnuplot's internal variables

There are two important properties of internal variables:

+ *Internal variables are only created when needed*.
+ *Internal variables are read-only*.

## 5.6 Inspecting file contents with the stats command

With gnuplot's `stats` command, you can examine the contents of a file and obtain
information about the number of valid and invalid records, as well as find
summary statistics for each column.

The syntax of the `stats` command is similar to that of `plot`:

```gnuplot
stats "records" u 1:2
```

## 5.7 Command history

### 5.7.1 Redrawing a graph

You can add new data or functions to `replot` after using `plot`:

```gnuplot
plot sin(x)
replot cos(x)
```

The `refresh` command is like `replot`, except that it *does not read the data again*.
This is convenient if data was read from standard input or when the file may
have changed on disk.

### 5.7.2 The general history feature

If you want more control over the command history, you can use `history`. The `history`
command can be used three ways: to print all or parts of the command history,
to search the command history, or to reexecute a command:

```gnuplot
history [quiet] [{int:max}] ["{str: filename}" [append]]
history ? "{str:cmd}"
history ! [{int:pos} | "{str:cmd}"]
```

### 5.7.3 Restoring session defaults

At times you may want to remove all user-defined variables adn options
and restore your gnuplot session to a defined state. You can do this using `reset`.

If you also want to relaod your personal initialization file, then you must
use `reset session`.
