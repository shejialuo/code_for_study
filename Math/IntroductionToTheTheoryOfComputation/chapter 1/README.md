# Regular Languages

The real computers are complicated. Instead, we use an idealized
computer called a *computation model*. We begin with the simplest
model, called the *finite state machine* or *finite automaton*.

## 1. Finite Automata

### 1.1 Introduction

In beginning to describe the mathematical theory of finite automata,
we do so in the abstract, without reference to any particular
application. The following figure depicts a finite automaton called
$M_{1}$.

![A finite automaton that has three states](https://s2.loli.net/2022/01/09/KlZsGAkuR3UBIgd.png)

The figure is called the *state diagram* of $M_{1}$. It has three
*states*, labeled $q_{1}, q_{2}$ and $q_{3}$. The *start state*, $q_{1}$,
is indicated by the arrow pointing at it from nowhere. The *accept state*,
$q_{2}$, is the one with a double circle. The arrows going from one state
to another are called *transitions*.

when this automaton receives an input string such as 1101, it processes
that string and produces an output. The output is either *accept* or
*reject*.

### 1.2 Formal Definition of A Finite Automaton

A finite automaton has several parts. It has a set of states and rules
for going from one state to another, depending on the input symbol. It
has an input alphabet that indicates the allowed input symbols. It has
a start state and a set of accept states.

The formal definition says that a finite automaton is a list of those
five objects: set of states, input alphabet, rules for moving, start
state and a set of accept states.

A *finite automaton* is a 5-tuple $(Q,\sum, \delta, q_{0}, F)$, where

+ $Q$ is a finite set called the *states*.
+ $\sum$ is a finite set called the *alphabet*.
+ $\delta: (Q \times \sum) \to Q$ is the *transition function*.
+ $q_{0} \in Q$ is the *start state*.
+ $F \subseteq Q$ is the *set of accept states*.

Now we can mathematically describe $M_{1} = (Q,\sum, \delta, q_{0}, F)$, where

+ $Q = \{q_{1}, q_{2}, q_{3}\}$
+ $\sum = \{0,1\}$
+ $\delta$ just can be computed
+ $q_{1}$ is the start state
+ $F = \{q_{2}\}$

If $A$ is the set of all strings that machine $M$ accepts, we say that $A$ is
the *language of machine* $M$ and write $L(M) = A$. We say that $M$ *recognizes* $A$
or that $M$ *accepts* $A$.

A machine may accept several strings, but it always recognizes only one language,
the empty language $\empty$.

Describing a finite automaton by state diagram is not possible in some cases. That
may occur when the diagram would be too big to draw or if the description depends
on some unspecified parameter. In these cases, we resort to a formal description
to specify the machine.

### 1.3 Formal Definition of Computation

Let $M = (Q,\sum, \delta, q_{0}, F)$ be a finite automaton and let $w = w_{1}w_{2}\cdots w_{n}$
be a string where each $w_{i}$ is a member of the alphabet $\sum$. Then $M$ accepts $w$
if a sequence of states $r_{0},r_{1},\dots,r_{n}$ in $Q$ exists with three conditions:

+ $r_{0} = q_{0}$
+ $\delta(r_{i},w_{i + 1}) = r_{i + 1}$, for $i = 0, \dots, n - 1$ and
+ $r_{n} \in F$

A language is called a *regular language* if some finite automaton recognizes it.

### 1.4 The Regular Operations

We define three operations on languages, called the *regular operations*, and
use them to study properties of the regular languages.

Let $A$ and $B$ be languages. We define the regular operations *union*, *concatenation*, and *stars*:

+ **Union**: $A \cup B = \{x | x \in A \ \text{or} \ x \in B\}$
+ **Concatenation**: $A \circ B = \{xy | x \in A \ \text{and} \ y \in B\}$
+ **Star**: $A^{*} = \{x_{1}x_{2}\dots x_{k} | k \geq 0 \ \text{and each} \ x_{i} \in A\}$

Let $\mathcal{N} = \{1,2,3,\dots\}$ be the set of natural numbers. When we say that
$\mathcal{N}$ is *closed under multiplication*, we mean that for any $x$ and $y$
in $\mathcal{N}$, the product $x \times y$ also in $\mathcal{N}$.

Generally speaking, a collection of objects is **closed** under some operation if
applying that operation to members of the collection returns an object still in the
collection.

#### Theorem 1

The class of regular languages is closed under the union operation. In other words,
if $A_{1}$ and $A_{2}$ are regular languages, so is $A_{1} \cup A_{2}$.

#### Proof 1

Let

+ $M_{1}$ recognize $A_{1}$, where $M_{1} = (Q_{1},\sum, \delta_{1}, q_{1}, F_{1})$
+ $M_{2}$ recognize $A_{2}$, where $M_{2} = (Q_{2},\sum, \delta_{2}, q_{2}, F_{2})$

Construct $M$ to recognize $A_{1} \cup A_{2}$, where $M = (Q,\sum, \delta, q_{0}, F)$.

1. $Q = \{(r_{1}, r_{2}) | r_{1} \in Q_{1} \ \text{and} \ r_{2} \in Q_{2} \}$. This set is
   the *Cartesian product* of sets $Q_{1}$ and $Q_{2}$.

2. $\sum$, the alphabet, is the same as in $M_{1}$ and $M_{2}$. In this theorem and in all subsequent
   similar theorems, we assume for simplicity that both $M_{1}$ and $M_{2}$ have the same input
   alphabet $\sum$. The theorem remains true if they have different alphabets, $\sum_{1}$ and
   $\sum_{2}$. We would then modify the proof to let $\sum = \sum_{1} \cup \sum_{2}$.

3. $\delta$, the transition function, is defined as follow. For each $(r_{1},r_{2}) \in Q$ and
   each $a \in \delta$, let

   $$
   \delta((r_{1},r_{2}), a) = (\delta_{1}(r_{1}, a),\delta_{2}(r_{2}, a))
   $$

4. $q_{0}$ is the pair $(q_{1}, q_{2})$.

5. $F$ is the set of pairs in which either member is an accept state of $M_{1}$ or $M_{2}$,
   we can write it as

   $$
   F = \{(r_{1},r_{2}) | r_{1} \in F_{1} \ \text{or} \ r_{2} \in F_{2} \}
   $$

#### Theorem 2

The class of regular languages is closed under the concatenation operation. In other words,
if $A_{1}$ and $A_{2}$ are regular languages then so is $A_{1} \circ A_{2}$.

## 2. Nondeterminism

Nondeterminism is a useful concept that has had great impact on the theory of computation.
So far in our discussion, every step of a computation follows in a unique way from the
preceding step. When the machine is in a given state and reads the next input symbol,
we know what the next state will be. We call this *deterministic* computation. In a
*nondeterministic* machine, several choices may exist for the next state at any point.

Nondeterministic is a generalization of determinism, so every deterministic finite
automaton is automatically a nondeterministic finite automaton

Nondeterminism may be viewed as a kind of parallel computation wherein multiple
independent "processes" or "threads" can be running concurrently. When the NFA splits
to follow several choices, that corresponds to a process "forking" into several children,
each proceeding separately. If at least one of these processes accepts, then the entire
computation accepts.

Another way to think of a nondeterministic computation is as a tree of possibilities.

Nondeterministic finite automata are useful in several respects:

+ Every NFA can be converted into an equivalent DFA.
+ Constructing NFAs is sometimes easier than directly constructing DFAs.
+ An NFA may be much smaller than its deterministic counterpart.
+ An NFA's functioning can be easier to understand.
