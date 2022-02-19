# Chapter 1 Prelude: understanding data with gnuplot

## 1.1 A busy weekend

It will be convenient to group the runners by completion time and to count the
number of participants who finished during each five-minute interval. The resulting
file might start like this:

```txt
# Minutes Runners
135      1
140      2
145      4
150      7
155      11
160      13
165      35
170      29
...
```

Gnuplot reads data from simple text files, with the arranged data in columns. To plot
a data file takes only single command, `plot`, like this:

```gnuplot
plot "marathon" using 1:2 with boxes
```

The `plot` command requires the name of data file as argument in quotes. By default,
gnuplot looks for the data file in the current working directory.

The rest of the command line specifies which columns to use for the plot and in which
way to represent the data. The `using 1:2` declaration tells gnuplot to use the first
and second columns in the file called `marathon`. The final part of the command,
`with boxes`, selects a box style.

## 1.2 What is graphical analysis?

1. Plot the data.
2. Inspect it, trying to find some recognizable behavior.
3. Compare the actual data to data that represents the hypothesis
4. Repeat
