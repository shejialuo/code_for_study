# Chapter 1 Introduction to Graph Models

## 1.1 Graphs and Digraphs

*DEFINITION*: A *graph* $G=(V,E)$ is a mathematical structure consisting
of two finite sets $V$ and $E$. The elements of $V$ are called *vertices*
(or *nodes*), and the element of $E$ are called *edges*. Each edge has
a set of one or two vertices associated to it, which are called *endpoints*.

*TERMINOLOGY*: An edge is said to *join* its endpoints. A vertex
joined by an edge to a vertex $v$ is said to be a *neighbor* of $v$.

*DEFINITION*: The (*open*) neighborhood of a vertex $v$ in a graph $G$,
denoted $N(v)$m is the set of all the neighbors of $v$. The *closes neighborhood* of
$v$ is given by $N[v] = N(v) \cup \{v\}$.

*NOTATION*: When $G$ is not the only graph under consideration, the notations
$V_{G}$ and $E_{G}$ (or $V(G)$ and $E(G)$) are used for the
vertex- and edge-sets of $G$.

### 1.1.1 Simple Graphs and General Graphs

*DEFINITION*: A *proper edge* is an edge that joins two distinct vertices.

*DEFINITION*: A *self-loop* is an edge that joins a single endpoint to itself.

*DEFINITION*: A *multi-edge* is a collection of two or more edges having identical end-points.
The *edge multiplicity* is the number of edges within the multi-edge.
*DEFINITION*: A *simple-graph* has neither self-loops nor multi-edges.

*DEFINITION*: A *loopless graph* (or *multi-graph*) may have multi-edges but no self-loops.

*DEFINITION*: A *(general) graph* may have self-loops and/or multi-edges.

*TERMINOLOGY*: When we use the term *graph* without a modifier, we mean a
*general graph*.

### 1.1.2 Null and Trivial Graphs

*DEFINITION*: A *null graph* is a graph whose vertex- and edge-sets are empty.

*DEFINITION*: A *trivial graph* is a graph consisting of one vertex and no edges.

### 1.1.3 Edge Directions

*DEFINITION*: A *directed edge* (or *arc*) is an edge, one of whose endpoints is
designated as the *tail*, and whose other endpoint is designated as the *head*.

*TERMINOLOGY*: An arc is said to be *directed from* its tail to its head.

*NOTATION*: In a general digraph, the head and tail of an arc $e$ may
be denoted $head(e)$ and $tail(e)$, respectively.

*DEFINITION*: Two arcs between a pair of vertices are said to be *oppositely directed*
if they do not have the same head and tail.

*DEFINITION*: A *multi-arc* is a set of two or more arcs having the same
tail and same head. The *arc multiplicity* is the number of arcs within the multi-arc.

*DEFINITION*: A *directed graph* (or *digraph*) is a graph each of whose edges
is directed.

*DEFINITION*: A digraph is *simple* if it has neither self-loops nor multi-arcs.

*NOTATION*: In a simple digraph, ar arc from vertex $u$ to vertex $v$ may
be denoted by $uv$ or by the ordered pair $[u,v]$.

*DEFINITION*: A *mixed graph* (or *partially directed graph*) is a
graph that has both undirected and directed edges.

*DEFINITION*: The *underlying graph* of a directed or mixed graph $G$
is the graph that results from removing all the designations of *head*
and *tail* from the directed edges of $G$.

### 1.1.4 Formal Specification of Graphs and Digraphs

*DEFINITION*: A *formal specification of a simple graph* is given by
an *adjacency table* with a row for each vertex, containing the list
of neighbors of that vertex.

*DEFINITION*: A *formal specification of a general graph* $G = (V, E, endpts)$
consists of a list of its vertices, a list of its edges, and
a two-row *incidence table* whose columns are indexed by the edges.
The same endpoint appears twice if $e$ is a self-loop.

![A general graph and its formal specification](https://s2.loli.net/2022/06/02/xbriEBjCWZqFwzI.png)

*DEFINITION*: A *formal specification of a general digraph* or
*a mixed graph* $D = (V,E,endpts,head,tail)$ is obtained from the
formal specification of the underlying graph by adding the functions
$head: E_{G} \to V_{G}$ and $tail: E_{G} \to V_{G}$, which designate
the $head$ vertex and $tail$ vertex of each arc.

![A general digraph and its formal specification](https://s2.loli.net/2022/06/02/kdevz9lKgT76hmL.png)

### 1.1.5 Degree of a Vertex

*DEFINITION*: *Adjacent vertices* are two vertices that are joined
by an edge.

*DEFINITION*: *Adjacent edges* are two distinct edges that have
an endpoint in common.

*DEFINITION*: If vertex $v$ is an endpoint of edge $e$, then $v$ is
said to be *incident* on e, and $e$ is incident on $v$.

*DEFINITION*: The *degree* (or *valence*) of a vertex $v$ in a graph $G$,
denoted $deg(v)$, is the number of proper edges incident on $v$
plus twice the number of self-loops.

*DEFINITION*: A vertex of degree $d$ is also called a $d$-valent vertex.

*NOTATION*: The smallest and largest degrees in a graph $G$ are denoted
$\delta_{min}$ and $\delta_{max}$.

*DEFINITION*: The *degree sequence* of a graph is the sequence formed by
arranging the vertex degrees in non-increasing order.

![A graph and its degree sequence](https://s2.loli.net/2022/06/02/h8wZmj93VQs56G2.png)

**Proposition 1.1.1**: A non-trivial simple graph $G$ have at least
one pair of vertices whose degrees are equal.

**Proof**: Suppose that the graph $G$ has $n$ vertices. Then there
appear to be $n$ possible degree values, namely $0,\dots, n-1$. However,
there cannot be both a vertex of degree 0 and a vertex of degree $n - 1$,
since the presence of a vertex of degree 0 implies that each of the
remaining $n - 1$ vertices is adjacent to at most $n-2$ other vertices.
Hence, the $n$ vertices of $G$ can realize at most $n-1$ possible
values for their degrees. Thus, the *pigeonhole principle* implies
that at least two of the $n$ vertices have equal degree.

**Theorem 1.1.2**: The sum of the degrees of the vertices of a graph
is twice the number of edges.

**Proof**: Each edge contributes two to the degree sum.

**Corollary 1.1.3**: In a graph, there is an even number of vertices
having odd degree.

**Proof**: Consider separately, the sum of the degrees that are odd and the
sum of those that are even. The combined sum is even by *Theorem 1.1.2*,
and since the sum of the even degrees is even, the sum of the odd
degrees must also be even. Hence, there must be an even number of
vertices of odd degree.

**Theorem 1.1.4**: Suppose that $\left<d_{1}, d_{2},\dots,d_{n}\right>$ is
a sequence of nonnegative integers whose sum is even. Then there exists
a graph with vertices $v_{1},v_{2},\dots,v_{n}$ such that
$deg(v_{i}) = d_{i}$, for $i = 1, \dots, n$.

**Proof**: Start with $n$ isolated vertices $v_{1}, v_{2}, \dots, v_{n}$.
For each $i$, if $d_{i}$ is even, draw $d_{i} / 2$ self-loops on vertex $v_{i}$,
and if $d_{i}$ is odd, draw $(d_{1} - 1) / 2$ self-loops. By
*Corollary 1.1.3*, there is an even number of odd $d_{i}$'s. Thus,
the construction can be completed by grouping the vertices associated with
the odd terms into pairs and the joining each pair by a single edge.

![Constructing a graph with degree sequence (5,4,3,3,2,1,0)](https://s2.loli.net/2022/06/02/YpcOw16sAVjyLlz.png)

### 1.1.6 Graphic Sequences

The construction in *Theorem 1.1.4* is straightforward but hinges on
allowing the graph to be non-simple. A more interesting problem is
determining when a sequence is the degree sequence of a *simple* graph.

*DEFINITION*: A non-increasing sequence $\left<d_{1},d_{2}, \dots, d_{n}\right>$ is
said to be *graphic* if it is the degree sequence of some sequence
of some simple graph. That simple graph is said to *realize* the sequence.

**Theorem 1.1.5**: Let $\left<d_{1},d_{2}, \dots, d_{n}\right>$
be a graphic sequence, with $d_{1} \geq d_{2} \geq \dots \geq d_{n}$.
Then there is simple graph with vertex-set $\{v_{1}, \dots, v_{n}\}$,
satisfying $deg(v_{i}) = d_{i}$ for $i = 1,2,\dots,n$, such that
$v_{1}$ is adjacent to vertices $v_{2}, \dots, v_{d_{1} + 1}$.

**Proof**: Among all simple graphs with vertex-set $\{v_{1},v_{2},\dots,v_{n}\}$
and $deg(v_{i})=d_{i}$, $i=1,2,\dots,n$, let $G$ be one for which
$r = \left|N_{G}(v_{1}) \cap \{v_{2}, \dots, v_{d_{1} + 1}\}\right|$
is maximum. If $r = d_{1}$, then the conclusion follows.
If $r < d_{1}$, then there exists a vertex $v_{s}$, $2 \leq s \leq d_{1} +  1$,
such that $v_{1}$ is not adjacent to $v_{s}$, and there exists
a vertex $v_{t}, t> d_{1} + 1$ such that $v_{1}$ is adjacent to $v_{t}$.
Moreover, since $deg(v_{s}) \geq deg(v_{t})$, there exists a vertex
$v_{k}$ such that $v_{k}$ is adjacent to $v_{s}$ but not to
$v_{t}$. Let $\tilde{G}$ be the graph obtained from $G$ by replacing
the edges $v_{1}v_{t}$ and $v_{s}v_{k}$ with the edges $v_{1}v_{s}$
and $v_{t}v_{k}$. Then the degrees are preserved. Thus,
$\left|N_{\tilde(G)}(v_{1}) \cap \{v_{2}, \dots, v_{d_{1} + 1}\}\right| = r + 1$,
which contradicts the choice of $G$ and completes the proof.

![Switching adjacencies while preserving all degrees](https://s2.loli.net/2022/06/02/zga39GrMDsbiYx6.png)

**Corollary 1.1.6**: A sequence $\left<d_{1},d_{2}, \dots, d_{n}\right>$ of
nonnegative integers such that $d_{1} \leq n - 1$ and
$d_{1} \geq d_{2} \geq \dots \geq d_{n}$
is graphic if and only if the sequence
$\left<d_{2} - 1, \dots, d_{d_{1} + 1} - 1, d_{d_{1} + 2}, \dots, d_{n}\right>$
is graphic.

![Algorithm 1.1.1](https://s2.loli.net/2022/06/02/HpdRUB7Ih2wQg1Y.png)

### 1.1.7 Indegree and Outdegree in a Digraph

*DEFINITION*: The *indegree* of a vertex $v$ in a digraph is the number
of arcs directed to $v$; the *outdegree* of vertex $v$ is the number
of arcs directed from $v$. Each self-loop at $v$ counts one toward
the indegree of $v$ and one toward the out degree.

**Theorem 1.1.7**: In a digraph, the sum of the indegrees and
the sum of the outdegrees both equal the number of edges.

**Proof**: Each directed edge $e$ contributes one to the indegree at head
and one to the outdegree at tail.

## 1.2 Common Families of Graphs

### 1.2.1 Complete Graphs

*DEFINITION*: A *complete graph* is a simple graph such that every
pair of vertices is joined by an edge. Any Complete graph on $n$ vertices
is denoted by $K_{n}$.

<!-- TODO: add the picture -->
![The first five complete graphs]()

### 1.2.2 Bipartite Graphs

*DEFINITION*: A *bipartite* graph $G$ is a graph whose vertex-set $V$
can be partitioned into two subsets $U$ and $W$, such that each edge
of $G$ has one endpoint in $U$ and one endpoint in $W$. The pair
$U,W$ is called a (*vertex*) *bipartition* of $G$, and $U$
and $W$ are called the *bipartition subsets*.

<!-- TODO: add the picture -->
![Two bipartite graphs]()

**Proposition 1.2.1**: A bipartite graph cannot have any self-loops.

**Proof**: This is an immediate consequence of the definition.

*DEFINITION*: a *complete bipartite graph* is a simple bipartite graph
such that every vertex in one of the bipartition subsets is
joined to every vertex in the other bipartition subset. Any complete
bipartite graph that has $m$ vertices in one of its bipartition subsets
and $n$ vertices in the other is denoted $K_{m,n}$.

### 1.2.3 Regular Graphs

*DEFINITION*: A *regular* graph is a graph whose vertices all have
equal degree. A $k$-regular graph is regular graph whose common degree
is $k$.


**