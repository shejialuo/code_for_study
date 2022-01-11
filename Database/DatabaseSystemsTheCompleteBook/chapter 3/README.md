# Chapter 3 Design Theory for Relational Databases

## 0. Introduction

There is a well developed theory for relational databases: "dependencies," their
implications for what makes a good relational database schema, and what we can
do about a schema if it has flaws.

## 1. Functional Dependencies

There is a design theory for relations that lets us examine a design carefully
and make improvements based on a few simple principles.

### 1.1 Definition of Functional Dependency

A **functional dependency** (FD) on a relation $R$ is a statement of the form If
two tuples of $R$ agree on all of the attributes $A_{1}, A_{2}, \dots, A_{n}$
(i.e., the tuples have the same values in their respective components for each of
these attributes), then they must also agree on all of another list of attributes
$B_{1}, B_{2}, \dots, B_{m}$.

We write this FD formally as $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$
and say that

$$
A_{1},A_{2},\dots,A_{n} \  \mathbf{functionally} \ \mathbf{determine} \ B_{1},B_{2},\dots,B_{m}
$$

The following figure suggests what this FD tells us about any two tuples $t$ and
$u$ in the relation $R$.

![The effect of a functional dependency on two tuples](https://i.loli.net/2021/05/11/EpQ8BiWVdyFvIH2.png)

If we can be sure every instance of a relation $R$ will be one in which a given
FD is true, then we say that $R$ **satisfies** the FD. It is important to remember
that when we say that $R$ satisfies an FD $f$, we are asserting a constraint on $R$.

And one functional dependency $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots
B_{m}$ is equivalent to the set of FD:

$$
\begin{align}
A_{1}A_{2}\cdots A_{n} &\to B_{1} \\
A_{1}A_{2}\cdots A_{n} &\to B_{2} \\
&\dots \\
A_{1}A_{2}\cdots A_{n} &\to B_{m} \\
\end{align}
$$

### 1.2 Keys of Relations

We say a set of one or more attributes $\{A_{1},A_{2},\dots,A_{n}\}$ is a *key*
for a relation $R$ if:

+ Those attributes functionally determine all other attributes of the relation.
+ No proper subset of $\{A_{1},A_{2},\dots,A_{n}\}$ functionally determines all
other attributes of $R$, i.e., a key must be **minimal**.

When a key consists of single attribute $A$, we often say that $A$ is a key.

Sometimes a relation has more than one key. If so, it is common to designate one
of the keys as the *primary key*.

### 1.3 Superkeys

A set of attributes that contains a key is called a *superkey*, short for "superset
of a key". Thus, every key is a superkey.

## 2. Rules About Functional Dependencies

In this section, we shall learn how to reason about FD.

### 2.1 Reasoning About Functional Dependencies

If a relation $R(A,B,C)$ satisfies the FD $A \to B$ and $B \to C$, then we can
deduce that $R$ satisfies the FD $A \to C$.

Let the tuples agreeing on attribute $A$ be $(a,b_{1},c_{1})$ and $(a,b_{2},c_{2})$.
Since $R$ satisfies $A \to B$, so we have $b_{1} = b_{2}$. And since $B \to C$,
so we can get $c_{1} = c_{2}$. Thus, $A \to C$.

FD often can be presented in several different ways, without changing the set of
legal instances of the relation. We say:

+ Two sets of FD $S$ and $T$ are *equivalent* if the set of relation instances
satisfying $S$ is exactly the same as the set of relation instances satisfying $T$.
+ More generally, a set of FD $S$ follows from a set of FD $T$ if every relation
instance that satisfies all the FD in $T$ also satisfies all the FD in $S$.

### 2.2 The Splitting/Combining Rule

+ We can replace an FD $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$ by
a set of FD $A_{1}A_{2} \cdots A_{n} \to B_{i} $ for $i = 1, 2, \dots, m$. This
transformation we call the *splitting rule*.
+ We can replace a set of FD $A_{1}A_{2} \cdots A_{n} \to B_{i} $ for
$i = 1, 2,\dots, m$ by the single
FD $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$.
We call this transformation the *combining rule*.

### 2.3 Trivial Functional Dependencies

A constraint of any kind on a relation is said to be **trivial** if it holds for
every instance of the relation, regardless of what other constraints are assumed.

When the constraints are FD, it is easy to tell whether an FD is trivial. They
are the FD $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$ such that

$$
\{B_{1},B_{2}, \dots, B_{m}\} \subseteq \{A_{1}, A_{2},\dots,A_{n}\}
$$

There is an intermediate situation in which some, but not all, of the attributes
on the right side of an FD are also on the left. This FD is not trivial, but it
can be simplified by removing from the right side of an FD those attributes that
appear on the left. That is:

The FD $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$ is equivalent to

$$
A_{1}A_{2} \cdots A_{n} \to C_{1}C_{2} \cdots C_{k}, C_{i} \in B, C_{i}\notin A
$$
We call this rule, illustrated below, the **trivial-dependency rule**.

![The trivial-dependency rule](https://i.loli.net/2021/05/11/SGW2UdDqN1BFEvR.png)

### 2.4 Computing the Closure of Attributes

Suppose $\{A_{1},A_{2},\dots,A_{n}\}$ is a set of attributes and $S$ is a set of
FD. The *closure* of $\{A_{1},A_{2},\dots,A_{n}\}$ under the FD in $S$ is the
set of attributes $B$ such that every relation that satisfies all the FD in set
$S$ also satisfies $A_{1},A_{2},\cdots,A_{n} \to B$.

We denote the closure of a set of attributes $A_{1}A_{2} \cdots A_{n}$ by
$\{A_{1},A_{2},\dots,A_{n}\} ^ {+}$.
Note that $A_{1},A_{2},\dots,A_{n}$ are always in
$\{A_{1},A_{2},\dots,A_{n}\} ^ {+}$
because the FD $A_{1}A_{2} \cdots A_{n} \to A_{i}$ is trivial.

The following picture illustrates the closure process.

![Computing the closure of a set of attributes](https://i.loli.net/2021/06/07/TR2OfrvlQXbnsL4.png)

**Algorithm:** Closure of a Set of Attributes.

**INPUT:** A set of attributes $\{A_{1},A_{2},\dots,A_{n}\}$ and a set of FD $S$.

**OUTPUT:** The closure $\{A_{1},A_{2},\dots,A_{n}\} ^ {+}$.

1. If necessary, split the FD of $S$, so each FD in $S$ has a single attribute
on the right.
2. Let $X$ be a set of attributes that eventually will become the closure.
Initialize $X$ to be $\{A_{1},A_{2},\dots,A_{n}\}$.
3. Repeatedly search for some FD $B_{1}B_{2} \cdots B_{m} \to C$ such that all
of $B_{1},B_{2},\dots,B_{m}$ are in the set of attributes $X$, but $C$ is not.
Add $C$ to the set $X$ and repeat the search. Since $X$ can only grow, and the
number of attributes of any relation schema must be finite, eventually nothing
more can be added to $X$, and this step ends.
4. The set $X$, after no more attributes can be added to it, is the correct value.

However, we need to think the functionality of the closure. Personally, I think
the closure of a set of attributes is a way to find the FD of the original set
under the given set of FD $S$. The definition above is way too mathematical, in
my view, it is just a form of section 2.1.

### 2.5 The Transitive Rule

The transitive rule lets us cascade two FD:

If $A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$ and $B_{1}B_{2} \cdots
B_{m} \to C_{1}C_{2} \cdots C_{k}$ hold in relation $R$, then
$A_{1}A_{2} \cdots A_{n} \to C_{1}C_{2} \cdots C_{k}$ also holds in $R$.

### 2.6 Closing Sets of Functional Dependencies

Sometimes we have a choice of which FD we use to represent the full set of FD
for a relation. If we are given a set of FD $S$, then any set of FD equivalent to
$S$ is said to be a *basis* for $S$.

A minimal basis for a relation is a basis $B$ that satisfies three conditions:

+ All the FD in $B$ have singleton right sides.
+ If any FD is removed from $B$, the result is no longer a basis.
+ If for any FD in $B$ we remove one or more attributes from the left side of $F$,
the result is no longer a basis.

### 2.7 Projecting Functional Dependencies

Suppose we have a relation $R$ with set of FD $S$, and we project $R$ by computing
$R_{1} = \pi_{L}(R)$, for some list of attributes $R$. What FD hold in $R_{1}$?

The answer is obtained in principle by computing the *projection of functional dependencies*
$S$, which is all FD that:

+ Follow from $S$.
+ Involve only attributes of $R_{1}$.

**Algorithm:** Projecting a Set of Functional Dependencies.

**INPUT**: A relation $R$ and a second relation $R_{1}$ computed by the projection
$R_{1} = \pi_{L}(R)$. Also, a set of FD $S$ that hold in $R$.

**OUTPUT**: The set of FD that hold in $R_{1}$.

**METHOD**:

+ Let $T$ be the eventual output set of FD. Initially, $T$ is empty.
+ For each set of attributes $X$ that is a subset of the attributes of $R_{1}$,
compute $X^{+}$. Add to $T$ all nontrivial FD $X \to A$ such that $A$ is both in
$X^{+}$ and an attribute of $R_{1}$.
+ Now, $T$ is a basis for the FD that hold in $R_{1}$, but may not be a minimal
  basis. We may construct a minimal basis by modifying $T$ as follows:

  + If there is an FD $F$ in $T$ that follows from the other FD in $T$, remove
  $F$ from $T$.
  + Let $Y \to B$ be an FD in $T$, with at least two attributes in $Y$, and let
  $Z$ be $Y$ with one of its attributes removed. If $Z \to B$ follows the FD in
  $T$, then replace $Y \to B$ by $Z \to B$.
  + Repeat the above steps in all possible ways until no more changes to $T$ can
  be made.

However, the description of this algorithm is too way mathematical. As we mentioned
earlier, we compute closures to find all the corresponding FD. So we still compute
all the attributes' closures and reduce the redundant things.

## 3. Design of Relational Database Schemas

Careless selection of a relational database schema can lead to redundancy and
related anomalies.

### 3.1 Anomalies

Problems such as redundancy that occur when we try to cram too much into a single
relation are called *anomalies*.
The principal kinds of anomalies that we encounter are:

+ **Redundancy**. Information may be repeated unnecessarily in several tuples.
+ **Update Anomalies**. We may change information in one tuple but leave the same
information unchanged in another.
+ **Deletion Anomalies**. If a set of values becomes empty, we may lose other
information as a side effect.

### 3.2 Decomposing Relations

The accepted way to eliminate these anomalies is to *decompose* relations.
Decomposition of $R$ involves splitting the attributes of $R$ to make the schemas
of two new relations.

Given a relation $R(A_{1},A_{2},\dots,A_{n})$, we may decompose $R$ into two
relations $S(B_{1},B_{2},\dots, B_{m})$ and $T(C_{1},C_{2},\dots,C_{k})$ such that:

+ $\{A_{1},A_{2},\dots,A_{n}\}$ = $\{B_{1},B_{2},\dots,B_{m}\} \cup \{C_{1},C_{2},\dots,C_{k}\}$
+ $S = \pi_{B_{1},B_{2},\dots,B_{m}}(R)$
+ $T = \pi_{C_{1},C_{2},\dots,C_{k}}(R)$

### 3.3 Boyce-Codd Normal Form

The goal of decomposition is to replace a relation by several that do not exhibit
anomalies. There is, it turns out, a simple condition under which the anomalies
discussed above can be guaranteed not to exist. This condition is called
*Boyce-Codd normal form*, or BCNF.

+ A relation of $R$ is in BCNF if and only if: whenever there is a nontrivial FD
$A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$ for $R$, it is the case
that $\{A_{1},A_{2},\dots,A_{n}\}$ is a superkey for $R$.

That is, the left side of every nontrivial FD must be a superkey.

### 3.4 Decomposition into BCNF

By repeatedly choosing suitable decompositions, we can break any relation schema
into a collection of subsets of its attributes with the following important properties:

+ These subsets are the schemas of relations in BCNF.
+ The data in the original relation is represented faithfully by the data in the
relations that are the result of the decomposition.

The decomposition strategy we shall follow is to look for a nontrivial FD
$A_{1}A_{2} \cdots A_{n} \to B_{1}B_{2} \cdots B_{m}$ that violates BCNF. We
shall add to the right side as many attributes as are functionally determined
by $\{A_{1},A_{2},\dots,A_{n}\}$. This step is not mandatory, but it often reduces
the total amount of work done.

**Algorithm:** BCNF Decomposition Algorithm

**INPUT:** A relation $R_{0}$ with a set of functional dependencies $S_{0}$.

**OUTPUT:** A decomposition of $R_{0}$ into a collection of relations, all of
which are in BCNF.

**METHOD**: The following steps can be applied recursively to any relation $R$
and set of FD $S$. Initially, apply them with $R = R_{0}$ and $S = S_{0}$.

1. Check whether $R$ is in BCNF. If so, nothing more needs to be done. Return
$\{R\}$ as the answer.
2. If there are BCNF violations, let one be $X \to Y$, and compute $X^{+}$. Choose
$R_{1} = X^{+}$ as one relation schema and let $R_{2}$ have attributes $X$ and
those attributes of $R$ that are not in $X^{+}$.
3. To compute the sets of FD for $R_{1}$ and $R_{2}$; let these be $S_{1}$ and
$S_{2}$, respectively.
4. Recursively decompose $R_{1}$ and $R_{2}$ using this algorithm. Return the union
of the results of these decompositions.

## 4. Decomposition: The Good, Bad, and Ugly

However, decomposition can also have some bad consequences. In this section, we
shall consider three distinct properties we would like a decomposition to have:

+ *Elimination of Anomalies*.
+ *Recoverability of Information*.
+ *Preservation of Dependencies*.

### 4.1 Recovering Information from a Decomposition

Since we learned that every two-attribute relation is in BCNF, why did
we have to go through the trouble of the complicated Algorithm? Why not
just take any relation $R$ and decompose it into relations, each of whose
schemas is a pair of $R$'s attributes?

The answer is that the data in the decomposed relations, even if their
tuples were each the projection of a relation instance of $R$, might not
allow us to join the relations of the decomposition and get the instance
of $R$ back. If we do get $R$ back, then we say the decomposition has a
*lossless join*.

If we decompose a relation according to Algorithm, then the original
relation can be recovered exactly by the natural join.

### 4.2 The Chase Test for Lossless Join

Consider that we have decomposed relation $R$ into relations with sets
of attributes $S_{1}, S_{2},\dots, S_{k}$, is is true that

$$
\pi_{S_{1}}(R) \bowtie \pi_{S_{2}}(R) \bowtie \cdots \bowtie \pi_{S_{k}}(R) = R
$$

Three important things to remember are:

+ The natural join is associative and commutative. It does not matter
what order we join the projections; we shall get the same relation
as a result. In particular, the result is the set of tuples $t$ such
that for all $i = 1,2, \dots, k$, $t$ projected onto the set
of attributes $S_{i}$ is a tuple in $\pi_{S_{i}}(R)$.
+ Any tuple $t$ is surely in $\pi_{S_{1}}(R) \bowtie \pi_{S_{2}}(R) \bowtie \cdots \bowtie \pi_{S_{k}}(R)$.
The reason is that the projection of $t$ onto $S_{i}$ is surely in $pi_{S_{i}}(R)$ for
each $i$.
+ As a consequence, $\pi_{S_{1}}(R) \bowtie \pi_{S_{2}}(R) \bowtie \cdots \bowtie \pi_{S_{k}}(R) = R$ when
the FD's in $F$ hold for $R$ if and only if every tuple in the join is also in $R$.
That is, the membership test is all we need to verify that the decomposition has a lossless join.

The *chase* test for a lossless join is just an organized way to
see whether a tuple $t$ in $\pi_{S_{1}}(R) \bowtie \pi_{S_{2}}(R) \bowtie \cdots \bowtie \pi_{S_{k}}(R)$
can be proved, using the FD's in $F$, also to be a tuple in $R$.

We draw a picture of what we know, called a *tableau*. Assuming $R$ has attributes
$A,B,\dots$ we use $a,b,\dots$ for the components of $t$. For $t_{i}$, we use the
same letter as $t$ in the components that are in $S_{i}$, but we subscript the letter
with $i$ if the component is not in $i$.

Next, we "chase" the tableau by applying the FD's in $F$ to equate symbols in the
tableau whenever we can. If we discover that one of the rows is actually the same
as $t$, then we have proved that any tuple $t$ in the join of the projections was
actually a tuple of $R$.

## 5. Third Normal Form

We need to relax our BCNF requirement slightly, in order to allow the occasional
relation schema without our losing the ability to check the FD's. The relaxed condition
is called "third normal form".

### 5.1 Definition of Third Normal Form

A relation $R$ is in *third normal form* (3NF) if:

+ Whenever $A_{1}A_{2} \codts A_{n} \to B_{1}B_{2} \cdots B_{m}$ is a nontrivial FD,
either $\{A_{1},A_{2},\dots,A_{n}\}$ is a superkey, or those of $B_{1},B_{2},\dots,B_{m}$
that are not among the A's, are each a member of some key.

An attribute that is a member of some key is often said to be *prime*. Thus, the 3NF
condition can be stated as "for each nontrivial FD, either the left side is a superkey,
or the right side consists of prime attributes only."

### 5.2 The Synthesis Algorithm for 3NF Schemas

We can explain and justify how we decompose a relation $R$ into a set of relation such
that:

+ The relations of the decomposition are are in 3NF.
+ The decomposition has a lossless join.
+ The decomposition has the dependency-preservation property.

**Algorithm**: Synthesis of Third-Normal-Form Relations With a Lossless Join and
Dependency Preservation.

**Input**: A relation $R$ and a set $F$ of functional dependencies that hold for $R$.

**Output**: A decomposition of $R$ into a collection of relations, each of which is in
3NF. The decomposition has the lossless-join and dependency-preservation properties.

**Method**: Perform the following steps:

1. Find a minimal basis of $F$, say $G$.
2. For each functional dependency $X \to A$ in $G$, use $XA$ as the schema of one of
the relations in the decomposition.
3. If none of the relation schemas from step 2 is a superkey for $R$, add another
relation whose schema is a key for $R$.

## 6. Multivalued Dependencies

A "multivalued dependency" is an assertion that two attributes or sets of attributes
are independent of one another.

### 6.1 Definition of Multivalued Dependencies

<!-- TODO: Chapter 6 and Chapter 7 -->
