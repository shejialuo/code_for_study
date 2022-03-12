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

A *finite automaton* is a 5-tuple $(Q,\Sigma, \delta, q_{0}, F)$, where

+ $Q$ is a finite set called the *states*.
+ $\Sigma$ is a finite set called the *alphabet*.
+ $\delta: (Q \times \Sigma) \to Q$ is the *transition function*.
+ $q_{0} \in Q$ is the *start state*.
+ $F \subseteq Q$ is the *set of accept states*.

Now we can mathematically describe $M_{1} = (Q,\Sigma, \delta, q_{0}, F)$, where

+ $Q = \{q_{1}, q_{2}, q_{3}\}$
+ $\Sigma = \{0,1\}$
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

Let $M = (Q,\Sigma, \delta, q_{0}, F)$ be a finite automaton and let $w = w_{1}w_{2}\cdots w_{n}$
be a string where each $w_{i}$ is a member of the alphabet $\Sigma$. Then $M$ accepts $w$
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

+ $M_{1}$ recognize $A_{1}$, where $M_{1} = (Q_{1},\Sigma, \delta_{1}, q_{1}, F_{1})$
+ $M_{2}$ recognize $A_{2}$, where $M_{2} = (Q_{2},\Sigma, \delta_{2}, q_{2}, F_{2})$

Construct $M$ to recognize $A_{1} \cup A_{2}$, where $M = (Q,\Sigma, \delta, q_{0}, F)$.

1. $Q = \{(r_{1}, r_{2}) | r_{1} \in Q_{1} \ \text{and} \ r_{2} \in Q_{2} \}$. This set is
   the *Cartesian product* of sets $Q_{1}$ and $Q_{2}$.

2. $\Sigma$, the alphabet, is the same as in $M_{1}$ and $M_{2}$. In this theorem and in all subsequent
   similar theorems, we assume for simplicity that both $M_{1}$ and $M_{2}$ have the same input
   alphabet $\Sigma$. The theorem remains true if they have different alphabets, $\Sigma_{1}$ and
   $\Sigma_{2}$. We would then modify the proof to let $\Sigma = \Sigma_{1} \cup \Sigma_{2}$.

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

### 2.1 Formal Definition of A Nondeterministic Finite Automaton

The formal definition of a nondeterministic finite automaton is similar
to that of a deterministic finite automaton. Both have states, an input
alphabet, a transition function, a start state, and a collection of
accept states. However, they differ in one essential way: in the type of
transition function.

In an NFA, the transition function takes a state and an input symbol or
*the empty string* and produces *the set of possible next states*. In order
to write the formal definition, we need to set up some additional notation.
For any set $Q$, we write $\mathcal{P}(Q)$ to be the collection of all
subsets of $Q$. Here $\mathcal{P}(Q)$ is called the *power set* of $Q$ For
any alphabet $\Sigma$ we write $\Sigma_{\epsilon}$ to be $\Sigma \cup \{\epsilon\}$.

A *nondeterministic finite automaton* is a 5-tuple $(Q,\Sigma, \delta,q_{0},F)$ where

1. $Q$ is finite set of states,
2. $\Sigma$ is a finite alphabet,
3. $\delta: Q \times \Sigma_{\epsilon} \to \mathcal{P}(Q)$ is the transition function,
4. $q_{0} \in Q$ is the start state, and
5. $F \subseteq Q$ is the set of accept states.

The formal definition of computation for an NFA is similar to that for a DFA. Let
$N = (Q,\Sigma,\delta, q_{0}, F)$ be an NFA and $w$ a string over the alphabet $\Sigma$.
Then we say that $N$ *accepts* $w$ if we can write $w$ as $w=y_{1}y_{2}\cdots y_{m}$,
where each $y_{i}$ is a member of $\Sigma_{\epsilon}$ and a sequence of states
$r_{0}, r_{1}, \dots, r_{m}$ exists in $Q$ with three conditions:

1. $r_{0} = q_{0}$,
2. $r_{i+1} \in \delta(r_{i}, y_{i+1})$, for $i = 0, \dots, m - 1$ and
3. $r_{m} \in F$

### 2.2 Equivalence of NFAs and DFAs

We say that two machines are *equivalent* if they recognize the same language.

#### Theorem 3

Every nondeterministic finite automaton has an equivalent deterministic
finite automaton.

#### Proof 3

