# Chapter 5 Algebraic and Logical Query Languages

## 1. Relational Operations on Bags

In this section, we shall consider relations that are bags (multisets)
rather than sets. That is, we shall allow the same tuple to appear more
than once in a relation.

### 1.1 Why Bags?

Commercial DBMS's implement relations that are bags, rather than
sets. An important motivation for relations as bags is that some
relational operations are considerably more efficient if we use the bag
model.

+ To take the union of two relations as bags, we simply copy one
relation and add to the copy all the tuples of the other relation.
+ We can simply project each tuple and add it to the result.

### 1.2 Union, Intersection, and Difference of Bags

These three operations have new definitions for bags. Suppose that $R$
and $S$ are bags, and that tuple $t$ appears $n$ times in $R$ and $m$ times in $S$.
Note that either $n$ or $m$ can be 0. Then:

+ In the bag union $R \cup S$, tuple $t$ appears $n + m$ times.
+ In the bag intersection $R \cap S$, tuple $t$ appears $\min(n,m)$ times.
+ In the bag difference $R - S$, tuple $t$ appears $\max(0, n- m)$ times.

### 1.3 Projection of Bags

If $R$ is the bag and we compute the bag-projection $\pi_{A,B}(R)$.

### 1.4 Selection on Bags

To apply a selection to a bag, we apply the selection condition to each
tuple independently. As always with bags, we do not eliminate duplicate
tuples in the result.

### 1.5 Product of Bags

The rule for the Cartesian product of bags is the expected one. Each
tuple of one relation is paired with each tuple of the other, regardless
of whether it is a duplicate or not.

### 1.6 Joins of Bags

Joining bags presents no surprises.

## 2. Extended Operators of Relational Algebra

Languages such as SQL have several other operations that have proved
quite important in applications.

### 2.1 Duplicate Elimination

Sometimes, we need an operator that converts a bag to a set. For that
purpose, we use $\delta(R)$ to return the set consisting of one copy of every
tuple that appears one or more times in relation $R$.

### 2.2 Aggregation Operators

There are several operators that apply to sets or bags of numbers or
strings. These operators are used to summarize or "aggregate" the values
in one column of a relation, and thus are referred to as *aggregation*
operators. The standard operators of this type are:

+ $\text{SUM}$
+ $\text{AVG}$
+ $\text{MIN}$ and $\text{MAX}$
+ $\text{COUNT}$

#### 2.3 Grouping

Sometimes, we need to consider the tuples of a relation in groups,
corresponding to the value of one or more columns, and we aggregate
only within each group.

![A relation with imaginary division into groups](https://s2.loli.net/2021/12/22/rVUDK7oOpylEuTP.png)

#### 2.4 The Grouping Operator

The subscript used with the $\gamma$ operator is a list $L$ of elements, each of
which is either:

+ An attribute of the relation $R$ to which the $\gamma$ is applied;
this attribute is one of the attribute by which $R$ will be grouped.
This element is said to be a *grouping attribute*.
+ An aggregation operator applied to an attribute of the relation.
To provide a name for the attribute corresponding to this aggregation
in the result, an arrow and new name are appended to the aggregation.
The underlying attribute is said to be an *aggregated attribute*.

The relation returned by the expression $\gamma_{L}(R)$ is
constructed as follows:

+ Partition the tuples of $R$ into groups. Each group consists of all
tuples having one particular assignment of values to the grouping
attributes in the list $L$. If there are no grouping attributes, the
entire relation $R$ is one group.
+ For each group, produce one tuple consisting of:
  + The group attributes' values for that group and
  + The aggregations, over all tuples of that group, for the aggregated
    attributes on List $L$.

### 2.5 Extending the Projection Operator

Let us reconsider the projection operator $\pi_{L}(R)$. In the classical relational
algebra, $L$ is a list of attributes of $R$. We extend the projection operator
to allow it compute with components of tuples as well as choose components.
In *extended projection*, also denoted $\pi_{L}(R)$, projection lists can having
following kinds of elements:

+ A single attribute of $R$.
+ An expression $x \to y$, where $x$ and $y$ are names for attributes.
The element $x \to y$ in the list $L$ asks that we take the attributes
$x$ of $R$ and *rename* it.
+ An expression $E \to z$, where $E$ is an expression involving
attributes of $R$, constants, arithmetic operators, and string
operators, and $z$ is a new name for the attribute that results from
the calculation implied by $E$.

#### 2.6 The Sorting Operator

The expression $\tau_{L}(R)$, where $R$ is a relation and $L$ a list of some of $R$'s
attributes, is the relation $R$, but with the tuples of $R$ sorted in the
order indicated by $L$. If $L$ is the list $A_{1},A_{2},\dots,A_{n}$, then the tuples of
$R$ are sorted first by their values of attribute $A_{1}$.

#### 2.7 Outerjoins

A property of the join operator is that it is possible for certain tuples
to be "dangling"; that is, they fail to match any tuple of the other
relation in the common attributes. Dangling tuples do not have any trace
in the result of the join, so the join may not represent the data of the
original relations completely.

We shall consider the "natural" case first, where the join is on equated
values of all attributes in common to the two relations.
The *outerjoin* $R \overset{\circ}{\bowtie} S$ is formed by starting with $R \bowtie S$,
and adding any dangling tuples from $R$ and $S$, and adding any dangling
tuples from $R$ or $S$. The added tuples must be padded with a special
*null* symbol, $\perp$.

There are many variants of the basic outerjoin idea. The *left outerjoin*
$R \overset{\circ}{\bowtie}_{L} S$ is like the outerjoin, but only
dangling tuples of the left argument $R$ are padded with $\perp$ and
added to the result. The *right outerjoin* $R \overset{\circ}{\bowtie}_{R} S$
like the outerjoin, but only the dangling tuples of the right
argument $S$ are padded with $\perp$ and added to the result.

## 3. A Logic for Relations

As an alternative to abstract query languages based on algebra, one can
use a form of logic to express queries. The logic query language *Datalog*
consists of if-then rules.

### 3.1 Predicates and Atoms

Relations are presented in Datalog by *predicates*. Each predicates takes a
fixed number of arguments, and a predicate followed by its arguments is
called an *atom*.

In essence, a predicate is the name of a function that returns a boolean
value. If $R$ is a relation with $n$ attributes in some fixed order, then we
shall also use $R$ as the name of a predicate corresponding to this relation.
The atom $R(a_{1},a_{2},\dots,a_{n})$ has value `TRUE` if
$(a_{1},a_{2},\dots,a_{n})$ is a tuple of $R$; the atom has
value `FALSE` otherwise.

### 3.2 Arithmetic Atoms

There is another kind of atom that is important in Datalog, an *arithmetic
atom*. This kind of *atom* is a comparison between two arithmetic expressions,
for example $x < y$ or $x + 1 \geq y + 4 \times z$.

### 3.3 Datalog Rules and Queries

Operations similar to those of relational algebra are described in Datalog
by *rules*, which consist of

+ A relation atom called the *head*, followed by
+ The symbol $\leftarrow$, which we often read "if", followed by
+ A *body* consisting of one or more atoms, called *subgoals*, which
may be either relational or arithmetic. Subgoals are
connected by `AND`.

A *query* in Datalog is a collection of one or more rules. If there is
only relation that appears in the rule heads, then the value
of this relation is taken to be the answer to the query.

### 3.4 Meaning of Datalog Rules

We can imagine the variables of the rule ranging over all possible
values. Whenever these variables have values that together make all
the subgoals true, then we see what the value of the head is
for those variables, and we add the resulting tuple to the relation
whose predicate is in the head.

There are, however, restrictions that we must place on the way
variables are used in rules, so that the result of a rule is a finite
relation and so that rules with arithmetic subgoals or with negated subgoals
make intuitive sense. This condition, which we call the *safety* condition is:

Every variable that appears anywhere in the rule must appear in some
nonnegated, relational subgoal of the body.

### 3.5 Extensional and Intensional Predicates

It is useful to make the distinction between

+ *Extensional* predicates, which are predicates whose relations are
stored in a database, and
+ *Intensional* predicates, whose relations are computed by
applying one or more Datalog rules.

An EDB predicate can never appear in the head of a rule, although
it can appear in the body of a rule. IDB predicates can appear in either
the head or the body of rules.

## 4. Relational Algebra and Datalog

### 4.1 Boolean Operations

Let the schemas for the two relations be $R(A, B, C)$ and $S(A, B, C)$.

For $R \cup S$, we use the two rules:

$$
\begin{align*}
U(x, y, z) &\leftarrow R(x, y, z) \\
U(x, y, z) &\leftarrow S(x, y, z)
\end{align*}
$$

For $R \cap S$, we use the rule:

$$
I(x, y, z) \leftarrow R(x, y, z) \ \mathbf{AND} \ S(x, y, z)
$$

For $R - S$, we use the rule:

$$
D(x, y, z) \leftarrow R(x, y, z) \ \mathbf{AND NOT} \ S(x, y, z)
$$

### 4.2 Projection

To compute a projection of a relation $R$, we use one rule with
a single subgoal with predicate $R$. The arguments of this subgoal
are distinct variables, one for each attribute of the relation. The head
has an atom with arguments that are the variables corresponding to
the attributes in the projection list, in the desired order.

### 4.3 Selection

Selections can be somewhat more difficult to express in Datalog. The simple
case is when the selection condition is the AND of one or more
arithmetic comparisons. In that case, we create a rule with:

1. One relational subgoal for the relation upon which we are performing the
selection. This atom has distinct variables for each component, one for
each attribute of the relation.
2. For each comparison in the selection condition, an arithmetic subgoal
that is identical to this comparison.

### 4.4 Product

The product of two relations $R \times S$ can be expressed by
a single Datalog rule. This rule has two subgoals, one for $R$
and one for $S$. Each of these subgoals has distinct variables,
one for each attribute of $R$ or $S$.

### 4.5 Joins

We can take the natural join of two relations by a Datalog rule
that looks much like the rule for a product.
