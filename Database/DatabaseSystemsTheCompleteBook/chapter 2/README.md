# Chapter 2 The Relational Model of Data

## 1. An Overview of Data Models

We define some basic terminology and mention the most important data models.

### 1.1 What is a Data Model?

A data model is a notation for describing data or information. The description
generally consists of three parts:

+ Structure of the data.
+ Operations on the data.
+ Constraints on the data.

### 1.2 Important Data Models

There are two important data models of database systems:

+ The relation model, including object-relational extensions.
+ The semistructured-data model, including XML and related standards.

### 1.3 Other Data Models

A modern trend is to add object-oriented features to the relational model. There
are two effects of object-orientation on relations:

+ Values can have structure, rather than being elementary types such as integer
or string.
+ Relations can have associated methods.

In a sense, the extensions, called the **object-relational** model, are analogous
to the way structs in C were extended to objects in C++.

## 2. Basics of the Relational Model

The relational model gives us a single way to represent data: as a two-dimensional
table called a **relation**. The following figure is an example of a relation,
which we shall call `Movies`.

![The relation Movies](https://i.loli.net/2021/05/06/T3lfeEXNziqHwI2.png)

### 2.1 Attributes

The columns of a relation are named by **attributes**. For relation, the attributes
are `Movies`, `title`, `year`, `length` and `genre`. Attributes appear at the
tops of the columns.

### 2.2 Schemas

The name of a relation and the set of attributes for a relation is called the
**schema** for that relation. We show the schema for the relation with the relation
name followed by a parenthesized list of its attributes:

`Movies(title, year, length, genre)`.

The attributes in a relation schema are a set, not a list.

### 2.3 Tuples

The rows of a relation, other than the header row containing the attribute names,
are called **tuples**. A tuple has one **component** for each attribute of the
relation. And we use parentheses to surround the tuple.

For example: `(Gone With the Wind, 1939, 231, drama)`.

### 2.4 Domains

The relational model requires that each component of each tuple be atomic; that
is, it must be of some elementary type such as integer or string.

It is further assumed that associated with each attribute of a relation is a
**domain**, that is, a particular elementary type.

It is possible to include the domain, or data type, for each attribute in a
relation schema. We shall do so by appending a colon and a type after attributes.

For example: `Movies(title:string, year:integer, length:integer, genre:string)`.

### 2.5 Equivalent Representations of a Relation

Relations are sets of tuples, not lists of tuples. Thus the order in which the
tuples of a relation are presented is immaterial, they are all equivalent.

### 2.6 Relation Instances

A relation about movies is not static; rather, relations change over time. We
expect to insert tuples for new movies, as these appear.

We shall call a set of tuples for a given relation an **instance**. A conventional
database system maintains only one version of any relation: the set of tuples
that are in the relation "now". This instance of the relation is called the
**current instance**.

### 2.7 Key of Relations

A set of attributes forms a **key** for a relation if we do not allow two tuples
in a relation instance to have the same value in all the attributes of the key.

## 3. Defining a Relation Schema in SQL

There are two aspects to SQL:

+ The Data-Definition sublanguage for declaring database schemas.
+ The Data-Manipulation sublanguage for querying database and for modifying the database.

### 3.1 Relations in SQL

There are three kinds of relations in SQL:

+ Stored relations, which are called tables.
+ Views, which are relations defined by a computation. These relations are not
stored, but are constructed, in whole or in part, when needed.
+ Temporary tables, which are constructed by the SQL language processor when it
performs its job of executing queries and data modifications.

The SQL `CREATE TABLE` statement declares the schema for a stored relation. It
gives a name for the table, its attributes, and their data types. It also allows
us to declare a key, or even several keys, for a relation.

### 3.2 Data Types

All attributes must have a data type:

+ Character strings of fixed or varying length. The type `CHAR(n)` denotes a
fixed-length string of up to $n$ characters. `VARCHAR(n)` also denotes a string
of up to $n$ characters. Normally, a string is padded by trailing blanks if it
becomes the value of a component that is a fixed-length string of greater length.
+ Bit strings of fixed or varying length. The type `BIT(n)` denotes bit strings
of length $n$, while `BIT VARYING(n)` denotes bit string of length up to $n$.
+ The type `BOOLEAN` denotes an attribute whose value is logical: `TRUE`, `FALSE`
and `UNKNOWN`.
+ The type `INT` or `INTEGER` denotes typical integer values. The type `SHORTINT`
also denote integers.
+ Floating-point numbers can be represented in a variety of ways. We may use the
type `FLOAT` or `REAL` for typical floating-point numbers. A higher precision can
be obtained with the type `DOUBLE PRECISION`. SQL also has types that are real
numbers with a fixed decimal point. `DECIMAL(n,d)` and `NUMERIC(n,d)` allow values
that consist of $n$ decimal digits, with the decimal point assumed to be $d$
positions from the right.
+ Dates and times can be represented by the data types `DATE` AND `TIME`. These
values are essentially character strings of a special form.

### 3.3 Simple Table Declarations

The simplest form of declaration of a relation schema consists of the keywords
`CREATE TABLE` followed by the name of the relation and a parenthesized, comma-separated
list of the attribute names and their types.

```sql
CREATE TABLE Movies (
    title      CHAR(100),
    year       INT,
    length     INT,
    genre      CHAR(10),
    studioName CHAR(30),
    producer   INT
);
```

### 3.4 Modifying Relation Schemas

We can delete a relation $R$ by the SQL statement:

```sql
DROP TABLE R;
```

We can also modify the schema of an existing relation. These modifications are
done by a statement that begins with the keywords `ALTER TABLE` and the name of
the relation. We then have several options, the most important of which are:

+ `ADD` followed by an attribute name and its data type.
+ `DROP` followed by an attribute name.

### 3.5 Default Values

When we create or modify tuples, we sometimes do not have values for all components.
So we need to use **default value** to solve this problem.

In general, any place we declare an attribute and its data type, we may add the
keyword `DEFAULT` and an appropriate value.

### 3.6 Declaring Keys

There are two ways to declare an attribute or set of attributes to be a key in
the `CREATE TABLE` statement that defines a stored relation:

+ We may declare one attribute to be a key when that attributes is listed in the
relation schema.
+ We may add to the list of items declared in the schema and additional declaration
that says a particular attribute or set of attributes forms the key.

There are two declarations that may be used to indicate keyness:

+ `PRIMARY KEY`
+ `UNIQUE`

The effect of declaring a set of attributes $S$ to be a key for relation $R$ either
using `PRIMARY KEY` or `UNIQUE` is the following:

+ Two tuples in $R$ cannot agree on all of the attributes in set $S$, unless one
of them is `NULL`. Any attempt to insert or update a tuple that violates this
rule causes the DBMS to reject the action that caused the violation.

In addition, if `PRIMARY KEY` is used, then attributes in $S$ are not allowed to
have `NULL`. For example:

```sql
CREATE TABLE MovieStar(
    name CHAR(30) PRIMARY KEY,
    address VARCHAR(255),
    gender CHAR(1),
    birthdate DATE
);
```

Alternatively, we can use a separate definition of the key:

```sql
CREATE TABLE MovieStar(
    name CHAR(30),
    address VARCHAR(255),
    gender CHAR(1),
    birthdate DATE,
    PRIMARY KEY (name)
);
```

## 4. An Algebraic Query Language

In this section, we introduce the data-manipulation aspect of the relational model.

### 4.1 Why Do We Need a Special Query Language

The relational algebra is useful because it is **less** powerful than C or Java.

### 4.2 Overview of Relational Algebra

The operations of the traditional relational algebra fall into four broad classes:

+ The usual set operations: union, intersection, and difference.
+ Operations that remove parts of a relation.
+ Operations that combine the tuples of two relations.
+ An operation called "renaming" that does not affect the tuples of a relation,
but changes the relation schema.

We generally shall refer to expressions of relational algebra as **queries**.

### 4.3 Set Operations on Relations

When we apply set operations to relations, we need to put some conditions on
$R$ and $S$:

+ $R$ and $S$ must have schemas with identical sets of attributes and the type
for each attribute must be the same in $R$ and $S$.
+ Before we compute the set-theoretic union, intersection, or difference of sets
of tuples, the columns of $R$ and $S$ must be ordered so that the order of attributes
is the same for both relations.

### 4.4 Projection

The **projection** operator is used to produce from a relation $R$ a new relation
that has only some of $R$'s columns. The value of expression
$\pi_{A_{1},A_{2},\dots,A_{n}}(R)$ is a relation that has only the columns for
attributes $A_{1}, A_{2}, \dots, A_{n}$ of $R$.

### 4.5 Selection

The **selection** operator, applied to a relation $R$, produces a new relation
with a subset of $R$'s tuples. The tuples in the resulting relation are those
that satisfy some condition $C$ that involves the attributes of $R$. We denote
this operation $\sigma_{C}(R)$.

$C$ is a conditional expression of the type with which we are familiar from
conventional programming languages;

### 4.6 Cartesian Product

The **Cartesian product** of two sets $R$ and $S$ is the set of pairs that can
be formed by choosing the first element of the pair to be any element of $R$ and
the second any element of $S$. This product is denoted by $R \times S$.

![Two relations and their Cartesian product](https://i.loli.net/2021/05/11/y3wlFJ6r5z4Rd7L.png)

### 4.7 Natural Joins

More often than we want to take the product of two relations, we find a need to
**join** them by pairing only those tuples that match in some way. The simplest
sort of match is the **natural join** of the two relations $R$ and $S$, denoted
$R \bowtie S$, in which we pair only those tuples from $R$ and $S$ that agree in
whatever attributes are common to the schemas of $R$ and $S$.

More precisely, let $A_{1},A_{2},\dots,A_{n}$ be all the attributes that are in
both the schema of $R$ and the schema of $S$. Then a tuple $r$ from $R$ and a
tuple $s$ from $S$ are successfully paired if and only if $r$ and $s$ agree on
each of the attributes $A_{1},A_{2},\dots,A_{n}$.

![Joining tuples](https://i.loli.net/2021/05/11/WUuYRSX4FLV7Tdn.png)

### 4.8 Theta-Joins

The natural join forces us to pair tuples using one specific condition. The notation
for a theta-join of relations $R$ and $S$ based on condition $C$ is
$R \bowtie_{C} S$. The result of this operation is constructed as follows:

+ Take the product of $R$ and $S$.
+ Select from the product only those tuples that satisfy the condition $C$.

### 4.9 Combining Operations to Form Queries

Relation algebra, like all algebras, allow us to form expressions of arbitrary
complexity by applying operations to the result of other operations.

### 4.10 Naming and Renaming

In order to control the names of the attributes used for relations that are
constructed by applying relational-algebra operations, it is often convenient to
use an operator that explicitly renames relations. We shall use the operator
$\rho_{S(A_{1},A_{2},\dots,A_{n})}(R)$ to rename a relation $R$.

### 4.11 Relations Among Operations

Some of the operations that we have described can be expressed in terms of other
relational-algebra operations. For example, intersection can be expressed in
terms of set difference:

$$
R \cap S = R - (R - S)
$$

The two forms of join are also expressible in terms of other operations. Theta-join
can be expressed by product and selection:

$$
R \bowtie_{C} S = \sigma_{C} (R \times S)
$$

The natural join of $R$ and $S$ can be expressed by starting with the product
$R \times S$. We then apply the selection operator with a condition $C$ of the form

$$
R.A_{1} = S.A_{1} \ \mathbf{AND} \ R.A_{2} = S.A_{2} \ \mathbf{AND}  \ \cdots \ \mathbf{AND} \ R.A_{n} = S.A_{n}
$$

Let $L$ be the list of attributes in the schema of $R$ followed by those attributes
in the schema of $S$ that are not also in the schema of $R$. Then

$$
R \bowtie S = \pi_{L}(\sigma_{C} (R \times S))
$$

### 4.12 A Linear Notation for Algebraic Expressions

The notation we shall use for assignment statements is:

+ A relation name and parenthesized list of attributes for that relation. The
name `Answer` will be used conventionally for the result of the final step.
+ The assignment symbol `:=`.
+ Any algebraic expression on the right. We can choose to use only one operator
per assignment, in which case each interior node of the tree gets its own assignment
statement. However, it is also permissible to combine several algebraic operations
in one right side, if it is convenient to do so.

## 5. Constraints on Relations

We now take up the third important aspect of a data model: the ability to restrict
the data that may be stored in a database.

### 5.1 Relational Algebra as a Constraint Language

There are two ways in which we can use expressions of relational algebra to
express constraints.

+ If $R$ is an expression of relational algebra, then $R = \emptyset$ is a constraint
that says "The value of $R$ must be empty".
+ If $R$ and $S$ are expressions of relational algebra, then $R \subseteq S$ is
a constraint that says "Every tuple in the result of $R$ must also be in the result
of $S$."

These ways of expressing constraints are actually equivalent in what they can
express, but sometimes one or the other is clearer or more succinct. That is,
the constraints $R \subseteq S$ could just as well have been written $R - S = \emptyset$.

### 5.2 Referential Integrity Constraints

A common kind of constraint, called a **referential integrity constraint**,
asserts that a value appearing in one context also appears in another, related context.

In general, if we have any value $v$ as the component in attribute $A$ of some
tuple in one relation $R$, then because of our design intentions we may expect
that $v$ will appear in a particular component (say for attribute $B$) of some
tuple of another relation $S$. We can express this integrity constraint in relational
algebra as $\pi_{A}(R) \subseteq \pi_{B}(S)$.

### 5.3 Key Constraints

Here, we shall see how we can express algebraically the constraint that a certain
attribute or set of attributes is a key for a relation.

Suppose that `name` is the key for relation:

`MovieStar(name, address, gender, birthdate)`

The idea is that if we construct all pairs of `MovieStar` tuples $(t_1, t_2)$,
we must not find a pair that agree in the `name` component and disagree in the
`address` component. To construct the pairs we use a Cartesian product, and to
search for pairs that violate the condition we use a selection. We then assert
the constraint by equating the result to $\emptyset$.

### 5.4 Additional Constraint Examples

Suppose we wish to specify that the only legal values for the `gender` attribute
of `MovieStar` are `'F'` and `'M'`. We can express this constraint by:

$$
\sigma_{gender \neq 'F'\  \mathbf{AND}\  gender\neq 'M'}(\mathbf{MovieStar}) = \emptyset
$$
