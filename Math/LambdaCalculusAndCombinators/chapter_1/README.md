# The lambda-calculus

## 1.1 Introduction

The system presented in this chapter will be the pure one, which is
syntactically the simplest.

To motivate the $\lambda$-notation, consider the everyday mathematical
expression $x-y$. This can be though of as defining either a function
$f$ of $x$ or a function $g$ of $y$

$$
f(x) = x - y, g(y) = x - y
$$

And we can use another notation:

$$
f: x \mapsto x - y, g: \mapsto x - y
$$

Church's notation is a systematic way of constructing, for each expression
involving $x$, a notation for the corresponding function of $x$. Church
introduced $\lambda$ as an auxiliary symbol and wrote

$$
f = \lambda x . x - y, g = \lambda y . x - y
$$

These equations are clumsier than the originals, but do not be put off by
this; the $\lambda$-notation is principally for denoting higher-order functions,
not just functions of numbers, and for this it turns ot to be no worse
than others.

The $\lambda$-notation can be extended to functions of more than one variable.
For example, the expression $x - y` determines two functions $h$ and $k$ for
two variables:

$$
h(x, y) = x − y, k(y, x) = x − y
$$

These can be denoted by

$$
h = \lambda xy . x - y, k = \lambda yx . x - y
$$

However, we can avoid the need for a special notation for functions of several
variables by using functions whose values are not numbers but other functions.

$$
h^{*} = \lambda x . (\lambda y . x - y)
$$

From now on, *function* will mean *function of one variable* unless explicitly
stated otherwise.

**Definition ($\lambda$-terms)** Assume that there is given an infinite
sequence of expressions $v_{0}, v_{00}, v_{000}, \dots$ called *variables*,
and a finite, infinite or empty sequence of expressions called *atomic constants*,
different from the variables. (When the sequence of atomic constants is empty
, the system will be called *pure*, otherwise *applied*). The set of expressions
called *$\lambda$-terms* is defined inductively as follows:

+ all variables and atomic constants are $\lambda$-terms (called *atoms*).
+ if $M$ and $N$ are any $\lambda$-terms, then $MN$ is a $\lambda$-term (called an
*application*).
+ if $M$ is any $\lambda$-term and $x$ is any variable, then $\lambda x . M$ is a
$\lambda$-term (called an *abstraction*).

**Notation** *Capital letters* will denote arbitrary $\lambda$-terms in this
chapter. Letters $x$, $y$, $z$, $u$, $v$, $w$ will denote variables throughout
the book, and distinct letters will denote distinct variables unless stated
otherwise.

*Parentheses* will be omitted in such a way that, for example $MNPQ$ will denote
the term $(((MN)P)Q)$。 Other abbreviations will be

$$
\begin{align*}
\lambda x . PQ  \quad &\text{for} \quad (\lambda x. (PQ)) \\
\lambda x_{1}x_{2}\dots x_{n} . M \quad &\text{for} \quad (\lambda x_{1} . (\lambda
 x_{2} . (\dots (\lambda x_{n} . M) \dots)))
\end{align*}
$$

*Syntactic identity* of terms will be denoted by $\equiv$.

In general, if $M$ has been interpreted as a function or operator, then $(MN)$ is
interpreted as the result of applying $M$ to argument $N$.

A term $(\lambda x . M)$ represents the operator or function whose value at an
argument $N$ is calculated by substituting $N$ for $x$ in $M$.

## 1.2 Term-structure and substitution

**Definition** The *length* of a term $M$ (called $lgh(M)$) is the total number
of occurrences of atoms in $M$. In more detail:

+ $lgh(a) = 1$
+ $lgh(MN) = lgh(M) + lgh(N)$
+ $lgh(\lambda x . M) = 1 + lgh(M)$

**Definition** For $\lambda$-terms $P$ and $Q$, the relation $P$ *occurs in* $Q$
is defined by induction on $Q$, thus:

+ $P$ occurs in $P$
+ if $P$ occurs in $M$ or in $N$, then $P$ occurs in $(MN)$.
+ if $P$ occurs in $M$ or $P \equiv x$, then $P$ occurs in $(\lambda x . M)$.

**Definition (Scope)** For a particular occurrence of $\lambda x . M$ in a term
$P$, the occurrence of $M$ is called the *scope* of the occurrence of $\lambda x$
on the left.

**Definition (Free and bound variables)** An occurrence of a variable $x$ in a term
$P$ is called.

+ *bound* if it is in the scope of a $\lambda x$ in $P$.
+ *bound and binding* if and only if it is the $x$ in $\lambda x$.
+ *free* otherwise

If $x$ has at least one binding occurrence in $P$, we call $x$ a *bound variable* of
$P$. If $x$ has at least one free occurrence in $P$, we call $x$ a *free variable* of
$P$. The set of all free variables of $P$ is called.

$$
\text{FV}(P)
$$

A *closed term* is a term without any free variables.

**Definition (Substitution)** For any $M,N,x$ define $[N/x]M$ to be the result of
substituting $N$ for every free occurrence of $x$ in $M$, and changing bound variables
to avoid clashes. The precise definition is by induction on $M$, as follows:

+ $[N/x]x \equiv N$
+ $[N/x]a \equiv a$
+ $[N/x](PQ) = ([N/x]P[N/x]Q)$
+ $[N/x](\lambda x . P) \equiv \lambda x. P$
+ $[N/x](\lambda y . P) \equiv \lambda y . P, \quad x \notin \text{FV}(P)$
+ $[N/x](\lambda y . P) \equiv \lambda y . [N/x]P, \quad x \in \text{FV}(P), y \notin \text{FV}(N)$
+ $[N/x](\lambda y . P) \equiv \lambda z . [N/x][z/y]P, \quad x \in \text{FV}, y \in \text{FV}(N) $

**Definition (Change of bound variables)** Let a term $P$ contain an
occurrence of $\lambda x . M$, and let $y \notin \text{FV}(M)$. The
act of replacing this $\lambda x . M$ by $\lambda y . [y/x]M$
is called a *change of bound variable* or an $\alpha$-converts to $Q$,
or

$$
P \equiv_{\alpha} Q
$$

For example:

$$
\begin{align*}
\lambda x y . x (xy) &\equiv \lambda x. (\lambda y. x(xy)) \\
&\equiv_{\alpha} \lambda x . (\lambda v . x(xv)) \\
&\equiv_{\alpha} \lambda u . (\lambda v . u(uv)) \\
&\equiv \lambda uv . u(uv)
\end{align*}
$$

## 1.3 $\beta$-reduction

A term of form $(\lambda x . M)N$ represents an operator $\lambda x . M$ applied
to an argument $N$. In this informal representation of $\lambda x . M$, its value
when applied to $N$ is calculated by substituting $N$ for $x$
in $M$, so $(\lambda x . M) N$ can be 'simplified' to $[N/x]M$.

**Definition ($\beta$-contracting, $\beta$-reducing)**. Any term of form

$(\lambda x. M) N$ is called a $\beta$-redex and the corresponding term
$[N/x]M$ is called its *contractum*. If and only if a term $P$
contains an occurrence of $(\lambda x . M) N$ and we replace that
occurrence by $[N /x]M$, and the result is $P'$, we say have contracted
the redex-occurrence in $P$ and $P$ $\beta$-contracts to $P'$.

$$
P \rhd_{1\beta} P'
$$

If and only if $P$ can be changed to a term $Q$ by a finite series of
$\beta$-contractions and changes of bound variables, we say $P$ $\beta$-reduces to $Q$

$$
P \rhd_{\beta} Q
$$

**Definition** A term $Q$ which contains no $\beta$-redexes is called
a $\beta$-normal form. The class of all $\beta$-forms is called
$\beta$-nf or $\lambda \beta$=nf. If a term $P$ $\beta$-reduces
to a term $Q$ in $\beta$-nf, then $Q$ is called a $\beta$-normal form of $P$.
