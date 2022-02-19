# Chapter 2 Tutorial: essential gnuplot

## 2.1 Simple plots

Because gnuplot is a plotting program, it should come as no surprise that
the most important gnuplot command is `plot`. It can be used to plot both
functions and data (typically from a file).

### 2.1.1 Invoking gnuplot and first plots

Gnuplot is a *text-based* plotting program: you interact with it through
command-line-like syntax.

Probably the simplest plotting command you can issue is:

```gnuplot
plot sin(x)
```

Let's say you want to add more functions to plot together with the sine.

```gnuplot
plot sin(x), x, x - (x**3)/6
```

However, in this situation, the range of $y$ value is far too large and
you can define the range below:

```gnuplot
plot [][-2:2] sin(x), x, x - (x**3)/6
```

### 2.1.2 Plotting data from a file

Gnuplot reads data from text files. The data is expected to be *numerical*
and to be stored in the file in *whitespace-separated columns*. Lines
beginning with a hash mark (#) are considered to be comment lines and are
ignored. The following listing shows a typical data file containing the
share prices of two fictitious companies.

```txt
# Average PQR and XYZ stock price (in dollars per share) per calendar year
1975 49 162
1976 52 144
1977 67 140
1978 53 122
1979 67 125
1980 46 117
1981 60 116
1982 50 113
1983 66 96
1984 70 101
1985 91 93
1986 133 92
1987 127 95
1988 136 79
1989 154 78
1990 127 85
1991 147 71
1992 146 54
1993 133 51
1994 144 49
1995 158 43
```

Plotting data from a file is simple, you can type

```gnuplot
plot "prices"
```

Because data files typically contain many different data sets, you'll
usually want to *select the columns* to be used as $x$ and $y$ values.
This is done through the `using` directive to the `plot` command:

```gnuplot
plot "prices" using 1:2
```

If you want to plot the price of XYZ shares in the same plot, you can do
easily:

```gnuplot
plot "prices" using 1:2, "prices" using 1:3
```

By default, data points from a file are plotted using unconnected
symbols. So you need to tell gnuplot what *style* to use for the data.
You do so using the `with` directive. Many different styles are
available. Among the most useful are `with linespoints`, which plots
each data point as a symbol and also connects subsequent points, and
`with lines`, which just plots the connecting lines, omitting the
individual symbols:

```gnuplot
plot "prices" using 1:2 with lines, \
     "prices" using 1:3 with linespoints
```

This looks good, but it's not clear from the graph which line is which.
Gnuplot automatically provides a *key*, which shows a sample of line or
symbol type used for each data set together with a text description. You
can do much better by including a `title` for each data set as part of
the `plot` command:

```gnuplot
plot "prices" using 1:2 title "PQR" with lines, \
     "prices" using 1:3 title "XYZ" with linespoints
```

### 2.1.3 Abbreviations and defaults

Gnuplot offers two more features to the more proficient user:
*abbreviations* and *sensible defaults*. Any command and subcommand or
option can be abbreviated to the shortest, non-ambiguous form.

So we can use:

```gnuplot
plot "prices" u 1:2 w l, "prices" u 1:3 w lp
```

But this is still not the most compact form possible. Whenever part of a
command isn't given explicitly, gnuplot first tries to interpolate the
missing values with values the user provided; failing that, it falls back
to sensible defaults.

Whenever a filename is missing, the most recent filename is interpolated.
You can use this to abbreviate the previous command even further:

```gnuplot
plot "prices" u 1:2 w l, "" u 1:3 w lp
```

## 2.2 Saving commands and exporting graphics

There are two ways to save your work in gnuplot: you can *save* the
gnuplot commands used to generate a plot, so that you can regenerate the
plot at a later time. Or you can *export* the graph to a file in a
standard graphics file format.

### 2.1 Saving and loading commands

If you save to a file the commands you used to generate a plot, you can
later load them again and regenerate the plot where you left off.
Gnuplot commands can be saved to a file using the `save` command:

```gnuplot
save "graph.gp"
```

This saves the current values of all options, as well as the most recent
`plot` command, to the specified file. This file can later be loaded
again using the load command:

```gnuplot
load "graph.gp"
```

An alternative to `load` is the `call` command, which is similar to
`load` but takes up to nine additional parameters after the filename to
load. The parameters are available in the loaded file in the variables
`ARG1` through `ARG9`. The special variable `ARG0` is set to the
filename, and the variable `ARGC` holds the number of parameters supplied
to `call`.

Command files are plain text files, usually containing exactly one
command per line. Several commands can be combined on a single line by
separating them with a semicolon(;).

### 2.2 Exporting graphs

There are three ways to export graphs:

+ Using the GUI button
+ Using the clipboard
+ Using terminals

In gnuplot parlance, a terminal is a graphics capable output device.
Traditionally, this may have been a specific piece of hardware, but
today a gnuplot terminal is merely a reference to the underlying
graphics library.

Contemporary terminals come in two flavors: interactive and file-based.
Interactive terminals create the graphs that you seen on the screen.

But to export a graph to a file, you need to employ a file-based terminal
that can generate output in the desired output format.

All of this is straightforward. But there is one stumbling block:
exporting a graph via a file-based terminal *requires multiple steps*. In
order to export a graph with a gnuplot terminal, *several commands* must
be executed in proper sequence:

```gnuplot
plot exp(-x**2)
set terminal pngcairo
set output "graph.png"
replot
set terminal qt
set output
```

1. Begin with an arbitrary `plot` command
2. Select the desired file-based terminal, using the `set terminal`.
3. Specify the name of the output file, using the `set output`.
4. Regenerate the last plot, this time sending it to the file-based
terminal and the named file.
5. Restore the interactive terminal again, using both `set terminal` and
`set output` command.

### 2.3 Managing options with set and show

Gnuplot has relatively few commands but a large number of *options*.
These options are used to control everything from the format of the
decimal point to the name of the output file.

The three commands used to manipulate individual options are as follows:

+ `show` displays the current value of an option
+ `set` changes the value of an option
+ `unset` disables a specific option or returns it to its default value

There's also a forth command, `reset`, which return *all* options to
their default values. The only options not affected by `reset` are those
directly influencing output generation: `terminal` and `output`.
