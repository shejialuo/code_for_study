# Chapter 4 Managing data sets and files

## 4.1 Quickstart: the standard data-file format

Data is usually fed to gnuplot as plain text in a simple,
whitespace-separated format.

Input files may also contain comments, text strings, and other
information, in addition to the data.

### 4.1.1 Comments and header lines

Lines beginning with a `#` are considered comment lines and are ignored.
You can also instruct gnuplot to ignore the first `n` lines of a file by
using the `skip` directive in the `plot` command:

```gnuplot
plot "data" skip 2 using 1:2
```

### 4.1.2 Selecting columns

You can pick out any column and plot its values against any other column
by means of the `using` directive. If no `using` directive is found,
gnuplot plots the first column against the line number.

## 4.2 Managing structured data sets

### 4.2.1 Multiple data sets per file: index

Here's a common scenario: monitoring web traffic across multiple hosts. A
script logs in to each host, pulls the log files for the last few days,
summarizes the hits per day, and writes the results to an output file
before moving on to the next host.

The important issue here is that the resulting file doesn't contain
individual data *points*, but entire data *sets*: one for each host.
Each data set spans several rows: one for each day(`traffic1`).

For gnuplot, blank lines in data file are significant. A
*single blank line* indicates a *discontinuity* in the data. The data
above and below the blank line is treated as belonging to the same
data set, but no connecting line is drawn between the records before
and after the blank.

In contrast, *double blank lines* are used to distinguish *data sets*
in the file. Each set can be addressed in the `plot` command as if it
were in a separate file using the `index` directive to `plot`.

The `index` directive follows immediately after the filename in the `plot`
syntax and takes at least one argument specifying which data
set to select from the file. The argument can be either
numeric or a string.

#### Selecting data sets by position

A numeric argument is treated as an index into an array, following
the C language convention of counting from 0. Therefore, to plot
only the traffic for the staging site, you can use

```gnuplot
plot "traffic1" index 1 using 1:2 w lp
```

The `index` directive can be abbreviated as `i`. It can take up to three arguments,
separated by colons:

```gnuplot
index {int:start}[:{int:end}][:{int:step}]
```

#### Selecting data sets by name

Gnuplot 5 offers a new way to pick out a single data set from a file:
if the data set is preceded by a comment line that includes the ID or name
of the data set, you can use this identifier in the `index` directive.

```gnuplot
plot "traffic1" index 1 using 1:2 w lp
plot "traffic1" index "host=staging" using 1:2 w lp
```

### 4.2.2 Records spanning multiple lines: the every directive

Consider the same arrangement of hosts as in the previous section,
together with an auxiliary script that gathers traffic data from
all hosts and dumps it into a single file(`traffic2`).

Here, each record for a single day spans three lines: one for each
host. If you want to plot the traffic for each host separately, you
can use the `every` directive to pick up only the relevant subset of
all lines:

```gnuplot
plot "traffic2" every 3::1 using 1:2 with lp
```

Using `every` directive, you can control how you step through individual
lines:

```gnuplot
every {int:step} [::{int:start}[::{int:end}]]
```

## 4.3 File format options in detail

You can control several aspects of the file format using options.

### 4.3.1 Number formats

Gnuplot can read both integers and floating-point numbers, as well
as numbers in *scientific notation*.

### 4.3.2 Comments

You can include comments in a data file on lines starting with the
comment character(`#`). The line must *start* with the comment and
is ignored entirely.

You can make gnuplot interpret additional characters as comment
characters by using the `set datafile commentschars` command.

```gnuplot
set datafile commentschars ["{str:chars}"]
```

### 4.3.3 Field separator

By default, fields (columns) are separated from one another by
whitespace, which means any number of space or tab characters.
You can change the field separator using the `set datafile separator`
command:

```gnuplot
set datafile separator [ "{str:char}" | whitespace | tab | comma ] 
```

Separator characters aren't interpreted as separators when inside quoted
strings: quoted strings are always interpreted as the entry of a single column.

### 4.3.4 Missing values

You can use the `set datafile missing` command to specify a string to
be used in a data file to denote missing data:

```gnuplot
set datafile missing ["{str:str}"]
```

Having an indicator for missing values is important when you're using a
whitespace-separated file format: if the missing value were left blank,
gnuplot won't recognize it as a column value and would use the value
from the *next* column instead.

### 4.3.5 Strings in data files

A valid text field can be any string of printable characters that doesn't
include blank spaces. If the string contains blanks, it must be enclosed
in *double* quotes to prevent gnuplot from interpreting the blanks as
column separators. (Single quotes don't work!)

## 4.4 Accessing columns and pseudocolumns

### 4.4.1 Accessing columns by position or name

Columns are usually accessed by their position in the file:

```gnuplot
plot "data" using 1:2
```

This is convenient for files with only a few columns, but it can also work
well for truly large files, because it's easy to iterate over numeric
column specifiers with gnuplot's *inline* looping feature.

```gnuplot
plot for [j=2:24] "data" u 1:j
```

You can also specify a column by *name* if the data file contains a set of
column labels in the first row of the file(`grains`).

```txt
Year  Wheat  Barley  Rye
1990 8 6 14
1991 10 5 12
1992 10 7 15
1993 11 5 13
1994 9 6 12
```

```gnuplot
plot "grains" skip 1 u 1:2
plot "grains" u 1: "Wheat"
plot "grains" u "Year": "Wheat"
```

Observe that the line containing the labels *must not be a comment line*.
Furthermore, if any of the entries in the `using` phrase are strings, the
entire first line is interpreted as labels and isn't included in the plot.

### 4.4.2 Pseudocolumns

Gnuplot supplies three pseudocolumns for each file it reads. They're numbered
0, -1 and -2.

The pseudocolumn 0 contains the line number in the current file, starting at
zero, without counting any comment, label, or skipped lines. You can also
access this column in inline transformations using `$0`. (This counter is
reset to zero when it encounters the double blank line)

The pseudocolumn -1 contains the line number, starting at zero, and is reset
by a single blank line.

The pseudocolumn -2 contains the index of the current data set within the data
file. When a double blank line is encountered in the file, the line number
resets to zero, and the index is incremented.

### 4.4.3 Column-access functions

You can use the `column()` function whenever an expression has become too
complicated for the `using` syntax, or in contexts where the `$` shorthand
isn't available.

## 4.5 Pseudofiles

Typically, the data rendered by the `plot` command is read from a file, but in
some situations it makes sense to accept data from a different source.

### 4.5.1 Reading data from standard input

When given the special filename `-`, gnuplot attempts to read data from standard input.

### 4.5.2 Heredocs

Gnuplot 5 contains a new feature that makes it possible to embed data in
a *command* file or to enter data in the command windows and
*store the data in a variable for the duration of the gnuplot session*.

#### Defining and using heredocs

You define a heredoc by preceding a variable identifier with a `$`,
followed by the `<<` redirection operator and an arbitrary sequence of characters
that will mark the end of the data section:

```gnuplot
$d << EOD
1 0.5
2 0.75
3 0.99
EOD
```

You can now use the heredoc as you'd use a file:

```gnuplot
plot $d using 1:2
```

Several gnuplot commands that usually write to a file can write to a heredoc, instead;
both `set print` and `set table` accept a heredoc identifier. This makes it possible
to have gnuplot commands "write" to a heredoc from within a gnuplot session.

A heredoc, once defined, occupies memory. To release these resources, use the `undefine` command.
Using `undefine $*` drops all currently existing heredocs.

### 4.5.3 Reading data from a subprocess

```gnuplot
plot "< a.out" using 1:2
```

### 4.5.4 Writing to a pipe

```gnuplot
set terminal pngcairo
set output "| convert - graph.gif"
plot sin(x)
set output
```

### 4.5.5 Generating data

Reading from the special filename `"+"` is like reading from a file
that contains exactly those x positions at which the function would be
evaluated, in a single column.
In other words, the following two commands are equivalent:

```gnuplot
plot sin(x)
plot "+" using 1:(sin($1)) w l
```

The special file `"++"` is the two-dimensional equivalent of "+".
