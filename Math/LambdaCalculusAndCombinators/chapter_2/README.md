# Combinatory logic

## 2.1 Introduction to CL

To motivate combinators, consider the commutative law of addition in
arithmetic, which says

$$
\forall x, y \quad x + y  = y + x
$$

The above expression contains bound variables $x$ and $y$. But these
can be removed, as follows. We first define an addition operation $A$ by

$$
A(x, y) = x + y
$$

And then introduce an operator $\mathbf{C}$ defined by

$$
(\mathbf{C}(f))(x,y) = f(y, x)
$$

Then the commutative law becomes simply

$$
A = \mathbf{C}(A)
$$

The operator $\mathbf{C}$ may be called a combinator; other examples
of such operators are the following:

+ $\mathbf{B}$ which composes two functions. $(\mathbf{B}(f,g))(x) = f(g(x))$
+ $\mathbf{I}$ the identifier operator $\mathbf{I}(f) =f$
+ $\mathbf{S}$ a stronger composition operator $(\mathbf{S}(f,g))(x) = f(x,g(x))$

**Definition (CL-terms)** Assume that there is given an infinite sequence
of expressions $v_{0},v_{00},v_{000}, \dots$ called *variables*, and a finite
or infinite sequence of expressions called *atomic constants*,
including three called *basic combinators*: $\mathbf{I}$, $\mathbf{K}$,
$\mathbf{S}$. The set of expressions called *CL-terms* is defined inductively as follows:

+ all variables and atomic constants are CL-terms.
+ if $X$ and $Y$ are CK-terms, then so is $(XY)$

**Definition** The length of $X$ (or $lgh(X)$ is the number of
occurrences of atoms in $X$:

+ $lgh(a) = 1$
+ $lgh(UV) = lgh(U) + lgh(V)$

**Definition** The relation $X$ occurs in $X$, or $X$ is a subterm
of $Y$, is defined thus:

+ $X$ occurs in $X$.
+ if $X$ occurs in $U$ or in $V$, then $X$ occurs in $(UV)$.

The set of all variables occurring in $Y$ is called $\text{FV}(Y)$.
(In CL-terms all occurrences of variables are free, because there
is no $\lambda$ to bind them).

**Definition (Substitution)** $[U/x]Y$ is defined to be the result of
substituting $U$ for every occurrence of $x$ in $Y$:

+ $[U/x]x \equiv U$
+ $[U/x]a \equiv a$
+ $[U/x](VW) \equiv ([U/x]V[U/x]W)$

## 2.2 Weak reduction

**Definition (Weak reduction)** Any term $\mathbf{I}X$, $\mathbf{K}XY$ or
$\mathbf{S}XYZ$ is called a *weak redex*. *Contracting* an occurrence
of a weak redex in a term $U$ means replacing one occurrence of

$$
\begin{align*}
\mathbf{I}X &\to X \\
\mathbf{K}XY &\to X \\
\mathbf{S}XYZ &\to XZ(YZ)
\end{align*}
$$

If and only if this change $U$ to $U'$, we say that $U$ *weakly contracts*
to $U'$, $U \rhd_{1w} U'$. If and only $V$ is obtained from $U$ by
a finite series of weak contractions, we say that $U$ (*weakly*) *reduces* to $V$:

$$
U \rhd_{w} V
$$

**Definition** A *weak normal form* is a term that contains no weak
redexes. If and only if a term $U$ weakly reduces to a weak normal
form $X$, we call $X$ a *weak normal form* of $U$.

## 2.3 Abstraction in CL

**Definition (Abstraction)** For every CL-term $M$ and every variable $x$,
a CL-term called $[x] . M$ is defined by induction on $M$, thus:

1. $[x] . M \equiv \mathbf{K}, x \notin \text{FV}(M)$
2. $[x] . x  \equiv \mathbf{I}$
3. $[x]. Ux \equiv U, x \notin \text{FV}(U)$
4. $[x].UV \equiv \mathbf{S}([x] . Y)([x] . V)$, if 1 nor 3 applies.
