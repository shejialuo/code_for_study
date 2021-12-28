# Chapter 6 The Database Language SQL

This chapter introduces the basics of SQL: the query language and database
modification statements.

Here we give schema example.

```txt
Movies(title, year, length, genre, studioName, producerC#)
StarsIn(movieTitle, movieYear, starName)
MovieStar(name, address, gender, birthdate)
MovieExec(name, address, cert#, netWorth)
Studio(name, address, presC#)
```

## 1. Simple Queries in SQL

Now, we can use SQL:

```sql
SELECT *
FROM Movies
WHERE studioName = 'Disney' AND year = 1990
```

### 1.1 Projection in SQL

```sql
SELECT title, length
FROM Movies
WHERE studioName = 'Disney' AND year = 1990
```

### 1.2 Selection in SQL

The selection operator of relational algebra, and much more, is available
through the `WHERE` clause of SQL.

+ `<>` is the symbol for "not equal to".
+ `=` is equality.

### 1.3 Comparison of Strings

Two strings are equal if they are the same sequence of characters.

### 1.4 Pattern Matching in SQL

SQL also provides the capacity to compare strings on the basis of a simple
pattern match. An alternative form of comparison expression is

```sql
s LIKE p
```

Where $s$ is a string and $p$ is a *pattern*, a string with two special characters
$\%$ and $\_$.

+ $\%$ in $p$ can match any sequence of 0 or more characters in $s$.
+ $\_$ in $p$ matches any one character in $s$.

### 1.5 Null values and Comparisons Involving NULL

SQL allows attributes to have a special value `NULL`, which is called the *null
value*. There are many different interpretations that can be put on null values.
Here are some of the most common:

+ *Value unknown*: "I know there is some value that belongs here but I don't
know what it is."
+ *Value inapplicable*: "There is no value that makes sense here."
+ *Value withheld*: "We are not entitled to know the value that belongs
here."

There are two important rules to remember when we operate upon a `NULL` value:

+ When we operate on a `NULL` and any value, including another `NULL`, using an
arithmetic operator like $\times$ or $+$, the result is `NULL`.
+ When we compare a `NULL` value and any value, including another `NULL`,
using a comparison operator like $=$ or $>$, the result is `UNKNOWN`.

### 1.6 The Truth-Value UNKNOWN

The rule is easy to remember if we think of `TRUE` as 1, `FALSE` as 0, and
`UNKNOWN` as 1/2. Then:

+ The `AND` of two truth-values is the minimum of those values.
+ The `or` of two truth-values is the maximum of those values.
+ The negation of truth value $v$ is $1-v$.

Only the tuples for which the condition has the value `TRUE` become part of the
answer.

### 1.7 Ordering the Output

To get output in sorted order, we may add to the select-from-where
statement a clause:

```sql
ORDER BY <list of attributes>
```

## 2. Query Involving More Than One Relation

Much of the power of relational algebra comes from its ability to combine
two or more relations through joins, products, unions, intersections, and
differences.

### 2.1 Products and Joins in SQL

SQL has a simple way to couple relations in one query: list each relation in the
`FROM` clause. Then, the `SELECT` and `WHERE` clauses can refer to the attributes
of any of the relations in the `FROM` clause.

```sql
SELECT name
FROM Movies, MovieExec
WHERE title = 'Star Wars' AND `producerC#` = `cert#`
```

### 2.2 Disambiguating Attributes

Sometimes we ask a query involving several relations, and among these relations
are two oe more attributes with the same name. SQL solves this problem by allowing
us to place a relation name and a dot in front of an attribute.

```sql
SELECT MovieStar.name, MovieExec.name
FROM MovieStar, MovieExec
WHERE MovieStar.address = MovieExec.address
```

### 2.3 Tuple Variables

Sometimes we need to ask a query that involves two or more tuples from the same relation.
We may list a relation $R$ as many times as we need to in the `FROM` clause,
but we need a way to refer to each occurrence of $R$. SQL allows us to define,
for each occurrence of $R$ in the `FROM` clause, an "alias" which we shall refer
to as a *tuple variable*. Each use of $R$ in the `FROM` clause is followed by
the (optional) keyword `AS` and the name of the tuple variable.

```sql
SELECT Star1.name, Star2.name
FROM MovieStar Start1, MovieStar Star2
WHERE Start1.address = Star2.address
  AND Star1.name < Star2.name
```

### 2.4 Interpreting Multirelation Queries

There are several ways to define the meaning of the select-from-where expressions
that we have just covered. All are *equivalent*, in the sense that they each give
the same answer for each query applied to the same relation instances.

#### 2.4.1 Nested Loops

The semantics that we have implicitly used in example so far is that of tuple
variables. Recall that a tuple variable ranges over all tuples of the corresponding
relation. A relation name that is not aliased is also a tuple variable ranging
over the iteration.

If there are several tuple variables, we may imagine nested loops, one for each
tuple variable, in which the variables each range over the tuples of their respective
relations. For each assignment of tuples to the tuple variables, we decide whether
the `WHERE` clause is true. If so, we produce a tuple consisting of the values
of the expressions following `SELECT`;

![Answering a simple SQL query](https://s2.loli.net/2021/12/20/uhGAN7aYkwTvOi5.png)

#### 2.4.2 Parallel Assignment

There is an equivalent definition in which we do not explicitly create nested
loops ranging over the tuple variables. Rather, we consider in arbitrary order,
or in parallel, all possible assignments of tuples from the appropriate relations
to the tuple variables. For each such assignment, we consider whether the `WHERE`
clause becomes true. Each assignment that produces a true `WHERE` clause contributes
a tuple to the answer; that tuple is constructed from the attributes of the
`SELECT` clause, evaluated according to that assignment.

#### 2.4.3 Conversion to Relational Algebra

A third approach is to relate the SQL query to relational algebra. We start
with the tuple variables in the `FROM` clause and take the Cartesian product
of their relations. If two tuple variables refer to the same relation, then this
relation appears twice in the product, and we rename its attributes so all
attributes have unique names.

### 2.5 Union, Intersection, and Difference of Queries

Sometimes we wish to combine relations using the set operations of relational
algebra: union, intersection, and difference. SQL provides corresponding operators
that apply to the results of queries, provided those queries produce relations
with the same list of attributes and attribute types. The keywords used are `UNION`,
`INTERSECT`, and `EXCEPT`.

## 3. Subqueries

In SQL, one query can be used in various ways to help in the evaluation of another.
A query that is part of another is called a *subquery*. There are a number of
other ways that subqueries can be used:

+ Subqueries can return a single constant, and this constant can be compared
with a `WHERE` clause.
+ Subqueries can return relations that can be used in various way in `WHERE` clause.
+ Subqueries can appear in `FROM` clauses, followed by a tuple variable that
represents the tuples in the result of the subquery.

### 3.1 Subqueries that Produce Scalar Values

An atomic value that can appear as one component of a tuple is referred to as
a *scalar*. In the above example, we have

```sql
SELECT name
FROM Movies, MovieExec
WHERE title = 'Star Wars' AND producerC# = cert#
```

There is another way to look at this query. We need the `Movies` relation only
t get the certificate number for the producer of `Star Wars`. Once we have it,
we can query the relation `MovieExec` to find the name of the person with this certificate.

```sql
SELECT name
FROM MovieExec
WHERE cert# =
  (SELECT producerC#
   FROM Movies
   where title = 'Star Wars');
```

### 3.2 Conditions Involving Relations

There are a number of SQL operators that we can apply to a relation $R$ and
produce a boolean result. However, the relation $R$ must be expressed as a
subquery. As a trick, if we want to apply these operators to a stored table `Foo`,
we can use the subquery `(SELECT * FROM Foo)`.

+ `EXISTS` $R$ is a condition that is true if and only if $R$ is not empty.
+ $s$ `IN` $R$ is true if and only if $s$ is equal to one of the values in $R$.
For similarity, $s$ `NOT IN` $R$.
+ $s > $ `ALL` $R$ is true if and only if $s$ is greater than every value in unary
relation $R$.
+ $s > $ `Any` $R$ is true if and only if $s$ is greater than at least one value
in unary relation $R$.

The `EXISTS`, `ALL`, `ANT` operators can be negated by putting `NOT` in front of
the entire expression.

### 3.3 Conditions Involving Tuples

A tuple in SQL is represented by a parenthesized list of scalar values. Examples
are `(123, 'foo')` and (name, address, networth). The first of these has constants
as components; The second has attributes as components. Mixing of constants and
attributes is permitted.

If a tuple $t$ has the same number of components as a relation $R$, then it makes
sense to compare $t$ and $R$ in section 3.2.

```sql
SELECT name
FROM MovieExec
WHERE cert# IN
  (SELECT producer#
   FROM Movies
   WHERE (title, year) IN
    (SELECT movieTitle, movieYear)
     FROM StarsIn
     WHERE starName = 'Harrison Ford');
```

However, it can be rewritten without nested subqueries.

### 3.4 Correlated Subqueries

The simplest subqueries can be evaluated once and for all and the result used
in a higher-level query. A more complicated use of nested subqueries requires
the subquery to be evaluated many times, once for each assignment of a value
to some term in the subquery that comes from a tuple variable outside the
subquery A subquery of this type is called a *correlated* subquery.

For example, we shall find the titles that have been used for two or more
movies We start with an outer query that looks at all tuples in the
relation.

```txt
Movies(title, year, length, genre, studioName, producerC#)
```

For each tuple, we ask in subquery whether there is a movie with the
same title and a greater year.

```sql
SELECT title
FROM Movies Old
WHERE year < ANY
      (SELECT year
       FROM Movies
       WHERE title = Old.title);
```

It may seem that we don't know what value `Old.title` has. However, as
we range over `Movies` tuples of the outer query, each tuple provides
a value of `Old.title`. We then execute the inner query.

When writing a correlated query it is important that we be aware of the
*scoping rules* for names. In general, an attribute in a subquery belongs
to one of the tuple variables in that subquery's `FROM` clause if some
tuple variable's relation has that attribute in its schema. If not, we
look at the immediately surrounding subquery, then to the one surrounding
that, and so on.

However, we can arrange for an attribute to belong to another tuple variable
if we prefix it by that tuple variable and a dot.

### 3.5 Subqueries in FROM Clauses

Another use for subqueries is as relations in a `FROM` clause. In a `FROM`
list, instead of a stored relation we may use a parenthesized subquery.
Since we don't have a name for the result of this subquery, we must give
it a tuple-variable alias.

### 3.6 SQL Join Expressions

We can construct relations by a number of variations on the join operator
applied to two relations. These variants include products, natural joins,
theta joins and outerjoins.

The simplest form of join expression is a *cross join*; that term is a
synonym for what we called a Cartesian product.

```sql
Movies CROSS JOIN StartsIn;
```

A more conventional theta join is obtained with the keyword `ON`. We put
`JOIN` between two relation names $R$ and $S$ and follow them by `ON` and
a condition.

```sql
Movies JOIN StarsIn ON
  title = movieTitle AND year = movieYear;
```

### 3.7 Natural Joins

A natural join differs from a theta join in that:

+ The join condition is that all pairs of attributes from the two relations
having a common name are equated, and there are no other conditions.
+ One of each pair of equated attributes is projected out.

The SQL natural join behaves exactly this way. Keywords `NATURAL JOIN`
appear between the relations to express the $\bowtie$ operator.

### 3.8 Outerjoins

The outerjoin operator is a way to augment the result of a join by the
dangling tuples, padded with null values. In SQL, we can specify an
outerjoin; `NULL` is used as the null value.

## 4. Full-Relation Operations

In this section we shall study some operations that act on relations as a whole,
rather than one tuples individually or in small numbers.

### 4.1 Eliminating Duplicates

Sometimes, we do not wish duplicates, then we may follow the keyword
`SELECT` by the keyword `DISTINCT`.

### 4.2 Duplicates in Unions, Intersections, and Differences

Unlike the `SELECT` statement, which preserves duplicates by default, the union, intersection, and
difference operations, normally eliminate duplicates.

In order to prevent the elimination of duplicates, we must follow the
operator `UNION`, `INTERSECT`, or `EXCEPT` by the keyword `ALL`.

### 4.3 Grouping and Aggregation in SQL

SQL provides all the capability of the $\gamma$ operator through
the use of aggregation operators in `SELECT` clauses and a special `GROUP BY` clause.

### 4.4 Aggregation Operators

SQL uses the five aggregation operators `SUM`, `AVG`, `MIN`, `MAX`, and `COUNT`.
These operators are used by applying them to a scalar-valued
expression, typically a column name, in a `SELECT` clause.
One exception is the expression `COUNT(*)` which counts all the
tuples in the relation that is constructed from the `FROM` clause and
`WHERE` clause of the query.

### 4.5 Grouping

To group tuples, we use a `GROUP BY` clause, following the `WHERE` clause.

For example:

```SQL
SELECT studioName, SUM(length)
FROM Movies
GROUP BY studioName;
```

The `SELECT` clause has two kinds of terms. These are the only terms
that may appear when there is an aggregation in the `SELECT` clause.

+ Aggregations.
+ Attributes.

### 4.6 Grouping, Aggregation, and Nulls

When tuples have nulls, there are a few rules we must remember:

+ The value `NULL` is ignored in any aggregation.
+ `NULL` is treated as an ordinary value when performing groups.
+ When we perform any aggregation except count over an empty bag of value,
the result is `NULL`. The count of an empty bag is 0.

## 5. Database Modifications

In this section, we shall focus on three types of statements that allow us to

+ Insert tuples into a relation.
+ Delete certain tuples from a relation.
+ Update values of certain components of certain existing tuples.

### 5.1 Insertion

The basic form of insertion statement is:
`INSERT INTO` $R_{A_{1}, \dots, A_{n}}$
`VALUES` $(v_{1}, \dots, v_{n})$.

A tuple is created using the value $v_{i}$ for
attribute $A_{i}$, for $i = 1,2,\dots,n$. If the
list of attributes does not include all attributes
of the relation $R$, then the tuple created has
default values for all missing attributes.

### 5.2 Deletion

The form of a deletion is `DELETE FROM` $R$ `WHERE <condition>`.

The effect of executing this statement is that every
tuple satisfying the condition will be deleted from
relation $R$.

### 5.3 Updates

While we might think of both insertions and deletion of tuples
as "updates" to the database, an *update* in SQL is a very specific kind
of change to the database: one or more tuples that already exist in
the database have some of their components changed.

The general form of an update statement is:
`UPDATE` $R$ `SET <new-value assignments> WHERE <condition>`.

## 6. Transactions in SQL

To this point, our model of operations on the database has
been that of one user querying or modifying the database. Now we need to consider concurrency.

### 6.1 Serializability

SQL allows the programmer to state that a certain transaction must
be *serializable* with respect to other transactions. That is,
these transactions must behave as if they were run *serially*.

### 6.2 Atomicity

In addition to nonserialized behavior that can occur if two or more
database operations are performed about the same time, it is
possible for a single operation to put the database in an
unacceptable state if there is a hardware or software "crash".

So combinations of database operations need to be done *atomically*;
that is, either they are both done or neither is done.

### 6.3 Transactions

A transaction is a collection of one or more operations on
the database that must be executed atomically and in a serializable manner.

SQL allows the programmer to group several statements into a
single transaction. The SQL command `START TRANSACTION` is used
to mark the beginning of a transaction. There are two ways to end a transaction:

+ The SQL statement `COMMIT` causes the transaction to end
successfully.
+ The SQL statement `ROLLBACK` causes the transaction to *abort*.

### 6.4 Read-Only Transactions

If we tell the SQL execution system that our current transaction is *read-only*,
that is, it will never change the database, then it is quite possible
that the SQL system will be able to take advantage of that knowledge. Generally
it will be possible for many read-only transactions accessing the same
data to run in parallel.

We tell the SQL system that the next transaction is read-only by:

```SQL
SET TRANSACTION READ ONLY;
```

### 6.5 Dirty Reads

A *dirty read* is a red of dirty data written by another transaction.
The risk in reading dirty data is that the transaction that wrote
it may eventually abort.

Sometimes the dirty read matters, and sometimes it doesn't. Other
times it matters little enough that it makes sense to risk an occasional
dirty read and thus avoid:

+ The time-consuming work by the DBMS that is needed to prevent
dirty reads, and
+ The loss of parallelism that results from waiting until there
is no possibility of a dirty read.

SQL allows us to specify that dirty reads are acceptable for a
given transaction. We use the `SET TRANSACTION` statement:

```SQL
SET TRANSACTION READ WRITE
  ISOLATION LEVEL READ UNCOMMITTED;
```

### 6.6 Other Isolation Levels

SQL provides a total of four *isolation levels*.

```SQL
SET TRANSACTION READ LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```
