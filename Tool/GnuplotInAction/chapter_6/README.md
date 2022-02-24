# Chapter 6 A catalog of styles

## 6.1 Why use different plot styles

The choice of plot style isn't just an aesthetic issue: different graphical
representations give a data set context and may even imply specific *semantics*:

+ Continuous lines indicate a function or a dense data set with little noise.
+ Explicit point symbols emphasize the discrete nature of a sparse.
data set and make individual data points stand out.
+ Boxes are mostly used for histograms and so imply counts of discrete events.
+ Error bars make the uncertainty in a data set explicit.

## 6.2 Styles and aspects

### 6.2.1 Chossing styles inline through with

Generally, plot styles are chosen *inline*, as part of the `plot` command by using `with` keyword.

### 6.2.2 The default sequence

For example:

```gnuplot
plot [0:4][-0.25:1.5] "sequence" u 1:2 w lp, "" u 1:3 w lp, "" u 1:4 w lp, -log(x)+(x-1) w l
```

All data sets are plotted using the same overall style, but gnuplot
makes sure to change the color of the lines used. Gnuplot also uses a
different point symbol for each data set.

## 6.3 A catalog of plotting styles

### 6.3.1 Core styles: lines and points

There are four styles I consider *core* styles, because they'are so
generally useful: `with points`, `with lines`, `with linespoints`, and `with dots`.

#### Points

The `points` style plots a small symbol for each data point. The symbols aren't
connected to each other. This is the default style for data.

The size of the symbol can be changed globally using the `set pointsize` command.
The parameter is a multiplier, defaulting to 1.0:

```gnuplot
set pointsize {flt:mult}
```

You can also change the `pointsize` (abbreviated `ps`) inline:

```gnuplot
plot "data" u 1:2 w points ps 3
```

#### Lines

The `lines` style doesn't plot individual data points, only straight lines connecting
adjacent points.

#### Linespoints

The `linespoints` style is a combination of the previous two.

### 6.3.2 Indicating uncertainly: styles with error bars or ranges

Sometimes you don't just want to show a single data point; you also want to
indicate a range with it.

#### Styles with error bars

There are two basic styles to show data with error bars in gnuplot: `errorbars` and
`errorlines`.

The appearance of both the `errorlines` and `errorbars` styles is determined
by the current line style.

The error bars themselves are drawn in the current line style. A tic mark
is placed at the ends of each error bar. You can control the size
of the tic mark using the `set bars` option:

```gnuplot
set bars [small | large | fullwidth | {flt:mult}]
```

#### Time-series styles

Gnuplot offers two styles that are mostly useful for time-series data:
`candlesticks` and `financebars`.

### 6.3.3 Styles with steps

Gnuplot offers three styles to generate steplike graphs consisting only of vertical
and horizontal lines. The only difference between the three styles is the
location of the vertical step:

+ `histeps` centers each bin around the supplied $x$ value.
+ `steps` places vertical step at the *end* of the bin
+ `fsteps` places the vertical step at the *front* of the bin.
