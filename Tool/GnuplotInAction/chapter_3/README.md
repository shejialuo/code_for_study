# Chapter 3 The heart of the matter: the plot command

## 3.1 Plotting functions and data

### 3.1.1 Plotting functions

To plot a function, you use the `plot` command followed by the function
to plot and any style directives you wish to use:

```gnuplot
plot sin(x)
plot cos(x) with linespoints
```

If you don't specify a style, gnuplot chooses a different line color or
dash pattern for each new function in the plot.

The `set samples` option controls the number of points at which a
function is evaluated to generate a plot. It defaults to 100 points, but
this isn't sufficient for curves.

### 3.1.2 Plotting data

To plot data, you again use the `plot` command, but in place of the
function to plot, you specify the name of the file containing the
data set. There is one additional wrinkle: you also need to specify
which columns from the file to select.

## 3.2 Math with gnuplot

### 3.2.1 Mathematical expressions

Gnuplot uses standard infix syntax for mathematical expressions,
including the normal operator for the four basic arithmetical
operations, as in most C-like languages. Parentheses can be used to
change the order of evaluation. Gnuplot has the exponentiation operator
(`**`).

It's important to understand that *integer division truncates*.

All the usual relational and logical operators are available as well,
including the ternary conditional operator (`?:`). A recent addition is
the *comma operator*, which can be used to evaluate several statements
as part of a single expression.

### 3.2.2 Built-in functions

Gnuplot provides all the mathematical functions you've come to expect on
any scientific calculator.

### 3.2.3 User-defined variables and functions

It's easy to define new variables by assigning an expression to a name.
The following listing shows several useful constants you may want to
define:

```gnuplot
e = 2.71828182845905
sqrt2 = sqrt(2.0)
```

Functions can be defined in a similar function, as shown next.

```gnuplot
f(x) = -x * log(x)
binom(n, k) = n!/(k!*(n-k)!)
```

Functions can have up to 12 variables and can contain other functions and
operators. You use them as you would any other function.

By default, gnuplot assumes that the independent dummy variable, which is
automatically replaced by a range of $x$ values when plotting a function,
is labeled $x$, but you can change this using the `set dummy` command.

All other variables occurring in a function must have been assigned
values before you can plot the function. For convenience, you can assign
values to parameters as part of the `plot` command.

```gnuplot
g(x) = exp(-0.5*(x/s)**2)/s
plot s=1 g(x), s=2 g(x), s=3 g(x)
```

But in general, it's better to make all parameters explicit *arguments*
in the function definition.

```gnuplot
g(x, s) = exp(-0.5*(x/s)**2)/s
plot g(x,1) g(x,2) g(x,3)
```

All functions and variables have global scope. There's no such thing as
a private variable for local scope.

You can obtain lists of all user-defined variables and functions using
the following two commands:

```gnuplot
show variables
show functions
```

### 3.2.4 Mathematically undefined values and NaN

Gnuplot tends to be pretty to tolerant when encountering undefined
values: rather than failing, it doesn't produce any graphical output for
data points with undefined values. For instance, you may safely plot the
square root over any plot range: if the argument is negative, gnuplot
won't fail, it just won't show anything on the graph for $x \leq 0$.

This behavior can be used to suppress data points or define piecewise
functions, usually in conjunction with the ternary conditional operator.

```gnuplot
f(x) = abs(x) < 1 ? 1 : NaN
```

## 3.3 Data transformations

Large-scale data transformations aren't what gnuplot is designed for.
Properly understood, this is one of gnuplot's main strengths: it does
a simple task and does it well, and it doesn't require learning an
entire toolset or programming language.

Nevertheless, gnuplot has the ability to perform arbitrary data
transformations on the fly as part of the `plot` command.

An arbitrary function can be applied to each data point as part of the
`using` directive in the `plot` command. If an argument to `using` is
enclosed in parentheses, it's treated not as a column number, but as
an expression to be evaluated. Inside the parentheses, you can access
the column values for the current record by preceding the column number
with a dollar sign(`$`).

## 3.4 Logarithmic plots

In gnuplot, it's easy to switch to and from logarithmic:

```gnuplot
set logscale
set logscale x
set logscale y
```
