# Chapter 8 Views and Indexes

We begin this chapter by introducing virtual views, which are relations that
are defined by a query over other relations. Virtual views are not stored in the database,
but can be queried as if they existed. The query processor will replace
the view by its definition in order to execute the query.

Views can also be materialized in the sense that they are constructed
periodically from the database and stored there. The existence of these
materialized views can speed up the execution of queries. A very important
specialized type of "materialized view" is the index, a stored data
structure whose sole purpose is to speed up the access to specified tuples
of one of the stored relations.

## 1. Virtual Views

Relations that are defined with a `CREATE TABLE` statement actually exist in the
database. That is, a SQL system stores tables in some physical organization.

There is another class of SQL relations, called *(virtual) views*, that do not
exist physically. Rather, they are defined by an expression much like a query.
Views, in turn, can be queried as if they existed physically, and in
some cases, we can even modify views.

### 1.1 Declaring Views

The simplest form of view definition is:

```sql
CREATE VIEW <view-name> AS <view-definition>;
```

Suppose we want to have a view that is a part of the

```txt
Movies(title, year, length, genre, studioName, producerC#)
```

We can define view by

```sql
CREATE VIEW paramountMovies AS
  SELECT title, year
  FROM Movies
  WHERE studioName = 'Paramount';
```

### 1.2 Querying Views

A view may be queried exactly as if it were a stored table.

```sql
SELECT title
FROM ParamountMovies
WHERE year = 1979;
```

It is also possible to write queries involving both views and base tables.

```sql
SELECT DISTINCT starName
FROM ParamountMovies, StartsIn
WHERE title = movieTitle AND year = movieYear;
```

The simplest way to interpret what a query involving virtual views means to replace
each view in a `FROM` clause by a subquery that is identical to the view definition.

```sql
SELECT DISTINCT starName
FROM (SELECT title, year
      FROM Movies
      WHERE studioName = 'Paramount'
    ) Pm, StartsIn
WHERE Pm.title = movieTitle AND Pm.year = movieYear;
```

### 1.3 Renaming

Sometimes, we might prefer to give a view's attributes names of our
own choosing, rather than use the names that come out of the query defining
the view. We may specify the attributes of the view by listing them,
surrounded by parentheses, after the name of the view in the `CREATE VIEW` statement:

```sql
CREATE VIEW MovieProd(movieTitle, prodName) AS
  SELECT title, name
  FROM Movies, MovieExec
  WHERE producerC# = cert#
```

## 2. Modifying Views

In limited circumstances it is possible to execute an insertion, deletion or
update to a view.

### 2.1 View Removal

An extreme modification of a view is to delete it altogether.

```sql
DROP VIEW ParamountMovies;
```

### 2.2 Updatable Views

SQL provides a formal definition of when modifications to a view are
permitted. The SQL rules are complex, but roughly, they permit modifications on views that
are defined by selecting some attributes from one relation $R$. Two important
technical points:

+ The `WHERE` clause must no involve $R$ in a subquery.
+ The `FROM` clause can only consist of one occurrence of $R$
and no other relation.
+ The list in the `SELECT` clause must include enough attributes that for
every tuple inserted into the view, we can fill the other attributes
out with `NULL` values or the proper default.

### 2.3 Instead-Of Triggers on Views

When a trigger is defined on a view, we can use `INSTEAD OF` in place of
`BEFORE` or `AFTER`. If we do so, then when an event awakens the trigger, the
action of the trigger is done instead of the event itself. That is, an
instead-of trigger intercepts attempts to modify the view and in its
place performs whatever action the database designer deems appropriate.

```sql
CREATE TRIGGER ParamountInsert
INSTEAD OF INSERT ON ParamountMovies
REFERENCING NEW ROW AS NewRow
FOR EACH ROW
INSERT INTO Movies(title, year, studioName)
VALUES(NewRow.title, NewRow.year, 'Paramount');
```

## 3. Indexes in SQL

An *index* on an attribute $A$ of a relation is a data structure that makes it
efficient to find those tuples that have a fixed value for attribute $A$.
We could think of the index as a binary search tree of (key,value) pairs, in
which a key $a$ is associated with a "value" that is the set of locations of
the tuples that have $a$ in the component for attribute $A$.
Such an index may help with queries in which the attribute $A$ is compared
with a constant, for instance $A = 3$, or even $A \leq 3$.

### 3.1 Motivation for Indexes

When relations are very large, it becomes expensive to scan all the tuples
of a relation to find those tuples that match a given condition.

```sql
SELECT *
FROM Movies
WHERE studioName = 'Disney' AND year = 1990;
```

There might be 10000 `Movies` tuple, of which only 200 were made in 1990.

The naive way to implement this query is to get all 10000 tuples and test
the condition of the `WHERE` clause on each. It would be more efficient if
we could obtain directly only the 10 or so tuples that satisfied both the
conditions of the `WHERE` clause.

### 3.2 Declaring Indexes

Although the creation of indexes is not part of any SQL standard up to and
including SQL-99, most commercial systems have a way for the database designer
to say that the system should create an index on a certain attribute for a
certain relation. The following syntax is typical. Suppose we want to have
an index on attribute `year` for the relation `Movies`. Then we say:

```sql
CREATE INDEX YearIndex ON Movies(year);
```

Since `title` and `year` form a key for `Movies`, we might expect it to be common
that for both these attributes will be specified:

```sql
CREATE INDEX KeyIndex ON Movies(title, year);
```

## 4 Selection of Indexes

Choosing which indexes to create requires the database designer to analyze
a trade-off. In practice, this choice is one of the principal factors that
influence whether a database design gives acceptable performance. Two important
factors to consider are:

+ The existence of an index on an attribute may speed up greatly
the execution of those queries in which a value, or range of values, is
specified for that attribute, and may speed up joins involving that attribute as well.
+ On the other hand, every index built for one or more attributes of some
relation makes insertions, deletions, and updates to that relation more
complex and time-consuming.

### 4.1 A Simple Cost Model

To understand how to choose indexes for a database, we first need to know where
the time is spent answering a query. The details of how relations are stored
will be taken up when we consider DBMS implementation. But for the
moment, let us state that the tuples of a relation are normally distributed among
many pages of a disk. One page, which is typically several thousand bytes at
least, will hold many tuples.

To example even one tuple requires that the whole page be brought into main
memory. On the other hand, it costs little more time to examine all the tuples
on a page than to example only one. There is a great time saving if the page
you want is already in main memory, but for simplicity we shall assume that
never to be the case, and every page we need must be retrieved from the disk.

### 4.2 Some Useful Indexes

Often, the most useful index we can put on a relation is an index on its key.
There are two reasons:

+ Queries in which a value for the key is specified are common. Thus,
an index on the key will get used frequently.
+ Since there is at most one tuple with a given key value, the index returns
either nothing or one location for a tuple. Thus, at most one page must be
retrieved to get that tuple into main memory.

When the index is not on a key, it may or may not be able to improve the
time spent retrieving from disk the tuples needed to answer a query. There
are two situations in which an index can be effective, even if it is not on a key:

+ If the attribute is almost a key.
+ If the tuples are "clustered" on that attribute.

### 4.3 Calculating the Best Indexes to Create

It might seem that the more indexes we create, the more likely it is that
an index useful for a given query will be available. However, if modifications are
the most frequent action, then we should be very conservative about
creating indexes. Each modification on a relation $R$ forces us to change any index
on one or more of the modified attributes of $R$. Thus, we must read and
write not only the pages of $R$ that are modified, but also read and write certain
pages that hold the index.

However, the indexes themselves have to be stored, at least partially, on disk, so accessing
and modifying the indexes themselves cost disk accesses.

To calculate the new value of an index, we need to make assumptions about which
queries and modifications are most likely to be performed on the database.
Sometimes, we have a history of queries that we can use to get good information,
on the assumption that the future will be like the past.

## 5. Materialized Views
