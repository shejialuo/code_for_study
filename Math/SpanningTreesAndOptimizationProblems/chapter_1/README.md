# Chapter 1 Spanning Trees

A spanning tree of a graph $G$ is a subgraph of $G$ that is a tree and
contains all the vertices of $G$. Spanning tree prove important for
several reasons:

+ They create a sparse subgraph that reflects a lot about the original
graph.
+ They play an important role in designing efficient routing algorithms.
+ Some computationally hard problems can be solved approximately by
using spanning trees.
+ They have wide applications in many areas.

## 1.1 Counting Spanning Trees

We use $n$ to denote the number of vertices of the input graph, and $m$
the number of edges of the input graph. Let's start with the problem
of counting the number of spanning trees. Let $K_{n}$ denote a complete
graph with $n$ vertices. How many spanning trees are there in the
complete graph $K_{n}$?

Back in 1889, Cayley devised the well-known formula $n^{n - 2}$ for the
number of spanning trees in the complete graph $K_{n}$. There are
numerous proofs of this elegant formula. The first explicit proof is
due to Prufer. The idea is to find a one-to-one corresponding between
the set of spanning trees of $K_{n}$.

**DEFINITION 1.1**: A Prufer sequence of length $n - 2$, for $n \geq 2$,
is any sequence of integers between 1 and $n$, with repetition allowed.

**LEMMA 1.1**: There are $n^{n-2}$ Prufer sequences of length $n-2$.

**PROOF**: By definition, there are $n$ ways to choose each element of
a Prufer sequence of length $n - 2$. Since there are $n - 2$ elements
to be determined, in total we have $n^{n-2}$ ways to choose the whole
sequence.

Given a labeled tree with vertices labeled by $1,2,3,\dots,n$, the
PRUFER ENCODING algorithm outputs a unique Prufer sequence of length
$n - 2$. It initializes with an empty sequence. If the tree has more
than two vertices, the algorithms finds the leaf with the lowest label,
and appends to the sequence the label of the neighbor of that leaf. Then
the leaf with the lowest label is deleted from the tree. This operation
is repeated $n - 2$ times until only two vertices remain in the tree.
The algorithm ends up deleting $n - 2$ vertices. Therefore, the resulting
sequence is of length $n - 2$.

**Algorithm**: PRUFER ENCODING

**Input**: A labeled tree with vertices labeled by $1,2,3,\dots, n$.

**Output**: A Prufer sequence

```pseudocode
Repeat n - 2 times
  v <- the leaf with lowest label
  Put the label of v's unique neighbor in the output sequence
  Remove v from the tree
```

It can be verified that different spanning trees of $K_{n}$ determine
different Prufer sequences. The PRUFER DECODING algorithm provides
the inverse algorithm, which finds the unique labeled tree $T$ with
$n$ vertices for a given Prufer sequence of $n - 2$ elements. Let the
given Prufer sequence be $P = (p_{1}, p_{2}, \dots, p_{n - 2})$.
Observe that any vertex $v$ of $T$ occurs $deg(v) - 1$ times in $P$.
Thus the vertices of degree one, the leaves are those that do not
appear in $P$. To reconstruct $T$ from $P$, we proceed as follows.
Let $V$ be the vertex label set $\{1,2,3,\dots, n\}$. In the $i^{th}$
iteration of the `for` loop, $P = (p_{i}, p_{i + 1}, \dots, p_{n - 2})$.
Let $v$ be the smallest element of the set $V$ that does not occur in
$P$. We connect vertex $v$ to vertex $p_{i}$. Then we remove $v$ from $V$,
and $p_{i}$ from $P$. Repeat this process for $n - 2$ times.

**Algorithm**: PRUFER DECODING

**Input**: A Prufer sequence $P = (p_{1}, p_{2}, \dots, p_{n - 2})$

**Output**: A labeled tree with vertices labeled by $1,2,3,\dots,n$

```pseudocode
P <- the input Pufer sequence
n <- |P| + 2
V <- {1,2,3,...,n}
Start with n isolated vertices labeled 1, 2, ..., n.
for i = 1 to n - 2 do
  v <- the smallest element of the set V that does not occur in P
  Connect vertex v to vertex pi
  Remove v from the set V
  Remove the element pi from the sequence P
Connect the vertices corresponding to the two numbers in V.
```

**THEOREM 1.1**: The number of spanning trees in $K_{n}$ is $n^{n-2}$.

In the following, we give a recursive formula for the number of spanning
trees in a general graph. Let $G - e$ denote the graph obtained by removing
edge $e$ from $G$. Let $G \ e$ denote the graph obtained by removing edge
$e$ from $G$. In other words, $G \ e$ is the graph obtained by deleting
$e$, and merging its ends. Let $\tau(G)$ denote the number of spanning trees
of $G$. The following recursive formula computes the number of spanning trees
in a graph.

**THEOREM 1.2**: $\tau(G) = \tau(G - e) + \tau(G \ e)$
