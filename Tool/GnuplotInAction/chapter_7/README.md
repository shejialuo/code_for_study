# Chapter 7 Decorations: labels, arrows, and explanations

## 7.1 Quick start: minimal context for data

The absolute quickest way to add the most important contextual information to
the plot is to give it a title, such as

```gnuplot
set title "Run Time (in seconds) vs. Cluster Size"
```

The title is centered at the top of the graph. With a little additional effort, you can put
labels on the axes using the `set xlabel` and `set y label` commands.

```gnuplot
set xlabel "Cluster size [thousands]"
set ylabel "Running time [sec]"
```

Finally, there's the key (or legend), which relates line types to data sets.
By default, it's placed in the top-right corner of the graph, but this may
not be suitable if it will interfere with data. You can change the position
of the key using the keywords `left, right, top, bottom, center` in
the `set key` command.

## 7.2 Understanding layers and locations

To add decorations to a plot, you must be able to tell gnuplot *where*
on the graph the new elements should be placed.

### 7.2.1 Locations

First, let's establish some terminology. The entire area of the plot is
referred to as the *screen* or the *canvas*. On the canvas is the *graph*,
which is surrounded by a *border*. The region outside the border is called
the *margin*.

![The parts of a gnuplot graph: canvas, borders, and margins](https://s2.loli.net/2022/02/24/M1ZDJYE3Bn7R2U5.png)

You can provide up two different sets of axes for a plot. This is occasionally
useful when comparing different data sets side by side: each data set can be
presented with its own coordinate system in a single graph. The primary
coordinate system (`first`) is plotted along the bottom and left
borders. The second coordinate system (`second`) is plotted along
the top and right borders.

Now that you know what all parts of a graph are called, we can talk about
the different ways to specify locations. Gnuplot uses *five* coordinate systems:

```gnuplot
first, second, graph, screen, character
```

The first two refer to the coordinates of the plot itself. The third and fourth
refer to the graph area and the entire canvas, respectively. Finally,
the `character` system gives positions in character widths and heights from
the origin $(0,0)$ of the screen area.

### 7.2.2 Layers

Gnuplot draws elements onto the canvas in a well-defined sequence. You can
use this stacking order to place some elements visually "in front of"
or "behind" other elements.

Plot elements are organized into *layers*. Graph elements in a layer
that's closer to the front obscure
elements in layers further back.

![Stacking order of layers](https://s2.loli.net/2022/02/24/TKweQfBnUuSPHkM.png)

Labels, arrows, and geometric shapes are organized in two layers, one of
which is behind the plotted data;
the other one is on the top of the visual stack. When creating a decoration, you can
assign it to either of these two layers using the keywords `front` and `back`;
the default is `back`. Within each layer, arrows are visually in front of labels,
and geometric shapes are even further behind.

A special layer, identified using the keyword `behind`, resides at
the bottom of the visual stack. Only objects can be added to this layer.
You can use this layer to provide a visual background to the entire graph.

## 7.3 Additional graph elements: decorations

### 7.3.1 Common conventions

Labels, arrows, and objects share some conventions that apply to all
of them equally.

#### Creating decorations

All decorations are created using the `set ...` command. It's very important
to remember that this command does not *generate* a replot event:
the decorations won't appear on the plot until the next command
has been issued.

#### Identifying decorations with tags

So that you can later refer to a specific object, you can give the object
a numeric tag: for instance, `set arrow 3...`. Arrows, labels, and objects
have separate counters.

If you omit the label, gnuplot will assign the next unused integer
automatically or apply the
command to *all* instances of the object.

### 7.3.2 Arrows

Arrows are generated using the `set arrow` command, which has the following options:

```gnuplot
set arrow [{idx:tag}] from {pos:from} to {pos:to}
set arrow [{idx:tag}] from {pos:from} rto {pos:offset}
set arrow [{idx:tag}] from {pos:from} length {pos:len} angle {flt:degrees}
set arrow [{idx:tag}] [nohead | head | heads | backhead]
          [size {flt:length} [, {flt:angle}], [, {flt:backangle}] [fixed]]
          [nofilled | empty | filled | noborder]
          [front | back]
          [lineoptions]
set arrow [{idx:tag}] [arrowstyle | as] {idx:style}
```

Each arrow has a beginning and an end point. The starting point
is always given explicitly, as a coordinate pair in any of gnuplot's coordinate systems,
but the end point can be given in three different ways:

+ As an explicit end point coordinate using the `to` keyword.
+ As a relative offset from the starting point, using `rto`.
+ As a length and an angle.

#### Customizing arrow appearance

You can customize the appearance of an arrow, in particular the size
and shape of its head and the look of its "backbone":

+ *Heads*.
+ *Head size and shape*.
+ *Head color and fill style*.
+ *Layer assignment*
+ *Line options*

![Different arrow forms and the commands used to generate them](https://s2.loli.net/2022/03/02/YwgLW5cVMHR7xJq.png)

### 7.3.3 Text labels

Text labels are a natural companion to arrows. The arrow shows
the observer where to look, and the label explains what's happening:

```gnuplot
set label [{idx:tag}] ["{str:text}"] [at {pos:location}]
                      [left | center | right]
                      [rotate [by {int:degrees}] | norotate]
                      [[no]boxed]
                      [front | back]
                      [point lt | pt | ps | nopoint]
                      [offset {pos:off}]
                      [hypertext]
                      [noenhanced]
                      [font, "{str:name}[, {int:size}]"]
```

### 7.3.4 Shapes or objects

The `set object ...` facility can place a variety of geometrical objects on a graph,
such as rectangles, circles, ellipses and even arbitrary polygons.

They introduce functionality to gnuplot that you'd expect in a general-purpose drawing program
rather than in a data-visualization tool.

All objects share a single tag counter, which is separate from the tag counters for
arrows and labels.

## 7.4 The graph's legend or key

The key (or legend) explains the meaning of each type of line or symbol placed
on the plot.

As usual, almost anything about the key can be configured.

```gnuplot
set key [on|off] [default]
        [[at {pos:position}]
         | [inside | outside] [lmargin | rmargin | tmargin | bmargin]]
        [left | right | center] [top | bottom | center]
        [vertical | horizontal ] [Left | Right]
        [ [no]reverse ] [[no]invert]
        [maxrows [auto | {int:rows}]] [maxcols [auto | {int:cols}]]
        [[no]opaque]
        [samplen [flt:len]] [spacing {flt:factor}]
        [width {int:chars} [height {int:chars}]]
        [[no]title "{str:text}"]
        [[no]box [lineoptions]]
        [[no]enhanced]
        [ font "str:font,int:size" ] [ textcolor | tc {clr:color} ]
        [ [no]autotitle [columnheader] ]
```
