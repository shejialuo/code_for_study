# Chapter 7 Constraints and Triggers

In this chapter we shall cover those aspects of SQL that let us create
"active" elements. An *active* element is an expression or statement
that we write once and store in the database, expecting the element to
execute at appropriate times.

One of the serious problems faced by writers of applications that update
the database is that the new information could be wrong in variety of
ways.

SQL provides a variety of techniques for expressing *integrity constraints*
as part of the database schema.

## 1. Keys and Foreign Keys

SQL allows us to define an attribute or attributes to be a key for a
relation with the keywords `PRIMARY KEY` or `UNIQUE`. SQL also uses
the term "key" in connection with certain referential-integrity constraints.
These constraints, called "foreign-key constraints", assert that a
value appearing in one relation must also appear in the primary-key
component of another relation.

### 1.1 Declaring Foreign-Key Constraints

In SQL we may declare an attribute or attributes of one relation to be
a *foreign key*, referencing some attribute(s) of a second relation. The
implication of this declaration is twofold:

+ The referenced attribute(s) of the second relation must be declared
`UNIQUE` or the `PRIMARY KEY` for their relation.
+ Values of the foreign key appearing in the first relation must also
appear in the referenced attributes of some tuple.

As for primary keys, we have two ways to declare a foreign key.

If the foreign key is a single attribute we may follow its name and
type by a declaration that it "references" some attributes of some table.

```SQL
REFERENCES <table>(<attribute>)
```

Alternatively, we may append to the list of attributes in a `CREATE TABLE`
statement one or more declarations stating that a set of attributes is
a foreign key. We then give the table and its attributes to which the
foreign key refers.

```SQL
FOREIGN KEY (<attributes>) REFERENCES <table>(<attributes>)
```

### 1.2 Maintaining Referential Integrity

There are three policies to maintain referential integrity:

+ *The Default Policy: Reject Violating Modifications*
+ *The Cascade Policy*. Under this policy, changes to the referenced
attribute(s) are mimicked at the foreign key.
+ *The Set-Null Policy*. Here, when a modification to the referenced
relation affects a foreign-key value, the latter is changed to `NULL`.

These options may be chosen for deletes and updates, independently, and
they are stated with the declaration of the foreign key. We declare them
with `ON DELETE` or `ON UPDATE` followed by our choice of `SET NULL` or
`CASCADE`.

### 1.3 Deferred Checking of Constraints

Because we need to maintain referential integrity. Sometimes, the
order we write SQL statements is important. However, there are cases
of *circular constraints* that cannot be fixed by judiciously ordering the
database modification steps we take.

The declaration of any constraint key, foreign-key, or other constraint
types may be followed by one of `DEFERRABLE` or `NOT DEFERRABLE`. The latter
is the default, and means that every time a database modification statement
is executed, the constraint is checked immediately afterwards, if the
modification could violate the foreign-key constraint. We follow
the keyword `DEFERRABLE` by either `INITIALLY DEFERRED` or `INITIALLY IMMEDIATE`. In
the former case, checking will be deferred to just before each
transaction commits. In the latter case, the check will be
made immediately after each statement.

There are two additional points about deferring constraints that we should
bear in mind:

+ Constraints of any type can be given names.
+ If a constraint has a name, say `MyConstraint`, then we can change
a deferrable constraint from immediate to deferred by the SQL statement

```sql
SET CONSTRAINT MyConstraint DEFERRED;
```

## 2 Constraints on Attributes and Tuples

Within a SQL `CREATE TABLE` statement, we can declare two kinds of
constraints:

+ A constraint on a single attribute.
+ A constraint on a tuple as a whole.

### 2.1 Not-Null Constraints

One simple constraint to associate with an attribute is `NOT NULL`. The
effect is to disallow tuples in which this attribute is `NULL`. The
constraint is declared by the keywords `NOT NULL` following the declaration
of the attribute in a `CREATE TABLE` statement.

### 2.2 Attribute-Based CHECK Constraints

More complex constraints can be attached to an attribute declaration by
the keyword `CHECK` and a parenthesized condition that must hold for
every value of this attribute.

An attribute-based `CHECK` constraint is checked whenever any tuple gets
a new value for this attribute. It is important to understand that an
attribute-based `CHECK` constraint is not checked if the database
modification does not change the attribute with which the constraint is
associated.

### 2.3 Tuple-Based CHECK Constraints

To declare a constraint on the tuples of a single table $R$, we may add
to the list of attributes and key or foreign-key declarations, in $R$'s
`CREATE TABLE` statement, the `CHECK` followed by a parenthesized condition.

## 3. Modification of Constraints

### 3.1 Giving Names to Constraints

In order to modify or delete an existing constraint, it is necessary that
the constraint have a name. To do so, we precede the constraint by the
keyword `CONSTRAINT` and a name for the constraint.

### 3.2 Altering Constraints on Tables

`ALTER TABLE` statements can affect constraints in several ways. You may
drop a constraint with keyword `DROP` and the name of the constraint to
be dropped. You may also add a constraint with the keyword `ADD`, followed
by the constraint to be added.

## 4. Assertions

The most powerful forms of active elements in SQL are not associated with
particular tuples or components of tuples. These elements, called "triggers"
and "assertions", are part of the database schema, on a pair with tables.

+ An assertion is a boolean-valued SQL expression that must be true at all times.
+ A trigger is a series of actions that are associated with certain events,
such as insertions into a particular relation, and that are performed
whenever these events arise.

### 4.1 Creating Assertions

The SQL standard proposes a simple form of *assertion* that allow us to
enforce any condition.

```SQL
CREATE ASSERTION <assertion-name> CHECK (<condition>)
```

### 4.2 Using Assertions

This is a difference between the way we write tuple-based `CHECK` constraints
and the way we write assertions. Tuple-based checks can refer directly to the attributes
of that relation in whose declaration they appear. An assertion has no such privilege.
Any attributes referred to in the condition must be introduced in the assertion,
typically by mentioning their relation in a select-from-where expression.

## 5. Triggers

*Triggers*, sometimes called *event-condition-action rules* or *ECA rules*,
differ from the kind of constraints discussed previously in three ways.

+ Triggers are only awakened when certain *events*, specified by the database programmer, occur.
The sorts of events allowed are usually insert, delete, or update to a particular relation.
Another kind of event allowed in many SQL systems is a transaction end.
+ Once awakened by its triggering event, the trigger test a *condition*.
+ If the condition of the trigger is satisfied, the *action* associated with the
trigger is performed by the DBMS.

### 5.1 Triggers in SQL

The SQL trigger statement gives the user a number of different options in the
event, condition, and action parts. Here are the principal features.

1. The check of the trigger's condition and the action of the trigger
may be executed either on the *state of the database* that exists
before the triggering event is itself executed or on the state that exists after
the triggering event is executed.
2. The condition and action can refer to both old and/or new values of
tuples that were updated in the trigger event.
3. It is possible to define update events that are limited to a particular
attribute or set of attributes.
4. The programmer has an option of specifying that the trigger executes either:
once for each modified or once dor all the tuples that are changed
in one SQL statement.

For example:

```sql
CREATE TRIGGER NetWorthTrigger
AFTER UPDATE OF netWorth ON MovieExec
REFERENCING
  OLD ROW AS OldTuple
  NEW ROW AS NewTuple
FOR EACH ROW
WHEN (OldTuple.netWorth > NewTuple.netWorth)
  UPDATE MovieExec
  SET netWorth = OldTuple.netWorth
  WHERE cert# = NewTuple.cert#
```

### 5.2 The Options for Trigger Design

+ We may replace `AFTER` by `BEFORE`, in which case the `WHEN` condition
is tested on the database state that exists before the triggering event is
executed.
+ Besides `UPDATE`, other possible triggering events are `INSERT` and `DELETE`.
+ The `WHEN` clause is optional. If it is missing, then the action is executed
whenever the trigger is awakened.
+ There can be any number trigger statements, separated by
semicolons and surrounded by `BEGIN...END`.
+ When `UPDATE`, we could use `OLD ROW AS` and `NEW ROW AS`. When `INSERT`,
we could only use `NEW ROW AS`. When `DELETE`, we could only use `OLD ROW AS`.
+ If we omit the `FOR EACH ROW` or replace it by the default `FOR EACH STATEMENT`, then
a row-level trigger becomes a statement-level trigger.
+ In a statement-level trigger, we cannot refer to old and new tuples
directly. However, any trigger can refer to the relation of *old tuples*
and the relation of *new tuples*, using declarations such as `OLD TABLE AS` and `NEW TABLE AS`.
