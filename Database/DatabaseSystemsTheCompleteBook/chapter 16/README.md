# Chapter 16 The Query Compiler

There are three broad steps that the query processor must take:

1. The query, written in a language like SQL, is *parsed*, that is,
turned into a parse tree representing the structure of the query in
a useful way.
2. The parse tree is transformed into an expression tree of relation algebra,
which we term a *logical query plan*.
3. The logical query plan must be turned into a *physical query plan*,
which indicates not only the operations performed, but the order in which
they are performed, the algorithm used to perform each step, and
the ways in which sorted data is obtained and data is passed from
one operation to another.

## 1. Parsing and Preprocessing

Omit.

## 2. Algebraic Laws for Improving Query Plans

### 2.1 Commutative and Associative Laws

Several of the operators of relational algebra are both associative
and commutative. Particularly:

+ $R \times S = S \times R$, $(R \times S) \times T$ = $R \times (S \times T)$.
+ $R \bowtie S = S \bowtie R$, $(R \bowtie S) \bowtie T = R \bowtie (S \bowtie T)$.
+ $R \cup S = S \cup R$, $(R \cup S) \cup T = R \cup (S \cup T)$.
+ $R \cap S = S \cap R$, $(R \cap S) \cap T = R \cap (S \cap T)$.

### 2.2 Laws Involving Selection

Since selections tend to reduce the size of relations markedly, one
of the most important rules of efficient query processing is
to move the selections down the tree as far as they will go without
changing the expression does.

To start, when the condition of a selection is complex, it helps to break
the condition into its constituent parts. The motivation is that
one part, involving fewer attributes than the whole condition, may
be moved to a convenient place where the entire condition cannot be evaluated.
Thus, our first two laws for $\sigma$ are the *splitting laws*:

+ $\sigma_{C_{1}} \ \text{AND} \ C_{2}(R) = \sigma_{C_{1}}(\sigma_{C_{2}}(R))$.
+ $\sigma_{C_{1}} \ \text{OR} \ C_{2}(R) = \sigma_{C_{1}}(R) \cup_{S} \sigma_{C_{2}}(R)$.

The next family of laws involving $\sigma$ allow us to push selections
through the binary operators: product, union, intersection, difference,
and join. There are three types of laws, depending on whether it is
optional or required to push the selection to each of the arguments:

1. For a union, the selection *must* be pushed to both arguments.
2. For a difference, the selection must be pushed to the first
argument and optionally may be pushed to the second.
3. For the other operators it is only require that the selection be pushed
to one argument.

+ $\sigma_{C}(R \cup S) = \sigma_{C}(R) \cup \sigma_{C}(S)$.
+ $\sigma_{C}(R - S) = \sigma_{C}(R) - S$ or $\sigma_{C}(R - S) = \sigma_{C}(R) - \sigma_{C}(S)$.

The next laws allow the selection to be pushed to one or both arguments.
If the selection is $\sigma_{C}$, then we can only push this selection
to a relation that has all the attributes mentioned in $C$, if there is one.
We shall show the laws below assuming that the relation $R$ has all the
attributes mentioned in $C$.

+ $\sigma_{C}(R \times C) = \sigma_{C}(R) \times S$.
+ $\sigma_{C}(R \bowtie S) = \sigma_{C}(R) \bowtie S$.
+ $\sigma_{C}(R \bowtie_{D} S) = \sigma_{C}(R) \bowtie_{D} S$.
+ $\sigma_{C}(R \cap S) = \sigma_{C}(R) \cap S$.

### 2.3 Pushing Selections

Pushing a selection down an expression tree, that is, replacing the left
side of one of the rules in Section 2.2 by its right side, is one of
the most powerful tools of query optimizers.

### 2.4 Laws Involving Projection

Projections, like selections, can be "pushed down" through many operators.
Pushing projects differs from pushing selections in what when we
push projects, it is quite usual for the projection also remain where
it is.

Pushing projections is useful, but generally less so than pushing selections.
The reason is that while selections often reduce the size of a relation
by a large factor, project keeps the number of tuples the same and only
reduces the length of tuples.

To describe the transformations of extended projection, we need to introduce
some terminology. Consider a term $E \to x$ on the list for a projection,
where $E$ is an attribute or an expression involving attributes
and constants. We say all attributes mentioned in $E$ are *input*
attributes of the projection, and $x$ is an *output* attribute.

The principle behind laws for projection is that:
We may introduce a projection anywhere in an expression tree,
as long as it eliminates only attributes that are neither used by
an operator above nor are in the result of the entire expression.

+ $\pi_{L}(R \bowtie S) = \pi_{L}(\pi_{M}(R) \bowtie \pi_{N}(S))$, where
$M$ and $N$ are the join attributes and the input attributes if
$L$ that re found among the attributes of $R$ and $S$.
+ $\pi_{L}(R \bowtie_{C} S) = \pi_{L}(\pi_{M}(R) \bowtie_{C} \pi_{N}(S))$
+ $\pi_{L}(R \times S) = \pi_{L}(\pi_{M}(R) \times \pi_{N}(S))$.

We can perform a projection entirely before a bag union. That is:

+ $\pi_{L}(R \cup_{B} S) = \pi_{L}(R) \cup_{B} \pi_{L}(S)$.

It is also possible to push a projection below a selection:

+ $\pi_{L}(\sigma_{C}(R)) = \pi_{L}(\sigma_{C}(\pi_{M}(R)))$.

### 2.5 Laws About Joins and Products

$$
R \bowtie_{C} S = \sigma_{C}(R \times S).
$$

### 2.6 Laws Involving Duplicate Elimination

The operator $\delta$, which eliminates duplicates from a bag, can be
pushed through many, but not all operators. In general, moving a $\delta$
down the tree reduces the size of intermediate relations and may
therefore be beneficial. Moreover, we can sometimes move the $\delta$
to a position where it can eliminated altogether.

+ $\delta(R) = R$ if $R$ has no duplicates. Important cases of such a
relation $R$ include
  + A stored relation with a declared primary key.
  + The result of a $\gamma$ operation, since grouping creates a relation with no duplicates
  + The result of a set union, intersection, or difference.
+ $\delta(R \times S) = \delta(R) \times \delta(S)$.
+ $\delta(R \bowtie S) = \delta(R) \bowtie \delta(S)$.
+ $\delta(R \bowtie_{C} S) = \delta(R) \bowtie_{C} \delta(S)$.
+ $\delta(R \cap_{B} S) = \delta(R) \cap_{B} S = R \cap_{B} \delta(S) = \delta(R) \cap_{B} \delta(S)$.

### 2.7 Laws Involving Grouping and Aggregation

+ $\delta(\gamma_{L}(R)) = \gamma_{L}(R)$.
+ $\gamma_{L}(R) = \gamma_{L}(\pi_{M}(R))$ if $M$ is a list containing at least
all those attributes of $R$ that are mentioned in $L$.
+ $\gamma_{L}(R) = \gamma_{L}(\delta(R))$ provided $\gamma_{L}$ is duplicate-impervious (`MIN`, `MAX`).

## 3. From Parse Trees to Logical Query Plans

### 3.1 Conversion to Relational Algebra

We shall now describe informally some rules for transforming SQL parse trees to
algebraic logical query plans. The first rule allows us to convert all "simple"
select-from-where to relational algebra directly. Its informal statement:

+ If we have a `<QUERY>` with a`<Condition>` that has no subqueries, then we replace
the entire construct by a relational-algebra expression consisting, from bottom
to top:
  + The product of all the relations mentioned in the `<FromList>`.
  + A selection $\sigma_{C}$, where $C$ is the `<Condition>` expression.
  + A project $\pi_{L}$, where $L$ is the list of attributes in the `<SelectList>`.

### 3.2 Removing Subqueries From Conditions

For parse trees with a `<Condition>` that has a subquery, we shall introduce an intermediate
form of operator, between the syntactic categories of the parse tree and the relational-algebra
operators that apply to relations. This operator is often called *two-argument selection*.
We shall represent a two-argument selection in a transformed parse tree by a
node labeled $\sigma$, with no parameter. Below this node is a left child that
represents the relation $R$ upon which the selection is being performed, and
a right child that is an expression for the condition applied to each tuple of $R$.

![An expression using two-argument selection](https://s2.loli.net/2023/05/26/rPEW3vtQu6BJxeY.png)

We need rules that allow us to replace a two-argument selection by a one-argument selection
and other operators of relational algebra. Each form of condition may require its own
rule. In common situations, it is possible to remove the two-argument selection
and reach an expression that is pure relational algebra. However, in extreme cases,
the two-argument selection can be left in place and considered part of the logical
query plan.

For example, we consider the condition in the above picture. Note that the subquery
in this condition is uncorrelated; that is, the subquery's relation can be
computed one and for all, independent of the tuple being tested. The rule of
eliminating such a condition is stated informally as follows:

+ Suppose we have a two-argument selection in which the first argument represents some
relation $R$ and the second argument is a `<Condition>` of the form $t$ `IN` $S$, where
expression $S$ is an uncorrelated subquery, and $t$ is a tuple composed of attributes
of $R$. We transform the tree as follows:

+ Replace the `<Condition>` by the tree that is the expression for $S$. If
$S$ may have duplicates, then it is necessary to include a $\delta$ operation
at the root of the expression of $S.
+ Replace the two-argument selection by a one-argument selection $\delta_{C}$,
where $C$ is the condition that equates each component of the tuple $t$ to the
corresponding attribute of the relation $S$.
+ Give $\delta_{C}$ an argument that is the product of $R$ and $S$.

![The transformation](https://s2.loli.net/2023/05/26/2JWcF3pwBzdbm1C.png)

The strategy for translating subqueries to relational algebra is more complex when
the subquery is correlated. We need to translate the subquery so that it produces
a relation in which certain extra attributes appear.

### 3.3 Improving the Logical Query Plan

When we convert our query to relation algebra we obtain one possible logical
query plan. The next step is to rewrite the plan using the algebraic laws
outlined. Alternatively, we could generate more than one logical plan, representing different
orders or combinations of operators. But in ths chapter, we shall assume that
the query rewriter chooses a single logical query plan that it believes
is "best".

We do, however, leave open the matter of what is known as "join ordering", so
a logical query plan that involves joining relations can be though of as
a family of plans, corresponding to the different ways a join could be ordered
and grouped.

There are a number of algebraic laws that tend to improve logical query plans. The
following are most commonly used in optimizers:

+ Selections can be pushed down the expression tree as far as they can go.
+ Projections can be pushed down the tree, or new projections can be added.
+ Duplicate eliminations can sometimes be removed, or moved to a more convenient position in the tree.

### 3.4 Grouping Associative/Commutative Operators

We shall perform a last step before producing the final logical query plan:
for each portion of the subtree that consists of nodes with the same associative
and commutative operator, we group the nodes with these operators into a single
node with many children.
