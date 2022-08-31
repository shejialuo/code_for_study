# Chapter 9 SQL in a Server Environment

## 1. The Three-Tier Architecture

There is a very common architecture for large database installations;
this architecture motivates the discussion of the entire chapter.
The architecture is called *three-tier* or *three-layer*.

+ *Web servers*.
+ *Application Servers*.
+ *Database Servers*.

## 2. The SQL Environment

### 2.1 Environments

A *SQL environment* is the framework under which data may exist
and SQL operations on data may be executed. In practice, we
should think of a SQL environment as a DBMS running at some installation.

All the database elements we have discussed are defined within a
SQL environment. These elements are organized into a hierarchy of structures,
each of which plays a distinct role in the organization.

Briefly, the organization consists of the following structures:

+ *Schemas*. These are collections of tables, views, assertions, triggers
and some other types of information.
+ *Catalogs*. These are collections of schemas.
+ *Clusters*. These are collections of catalogs. Each user has an
associated cluster.

### 2.2 Schemas

The simplest form of schema declaration is:

```sql
CREATE SCHEMA <schema name> <element declarations>
```

For example, we could declare a schema below.

```sql
CREATE SCHEMA MovieSchema
  CREATE TABLE MovieStar
  -- Create-table statements for the four other tables
  CREATE VIEW MovieProd
  -- Other view declarations
  CREATE ASSERTION RichPres;
```

It is not necessary to declare the schema all at once. One can modify
or add to the "current" schema using the appropriate `CREATE`, `DROP`, or `ALTER`
statement. We change the "current" schema with a `SET SCHEMA` statement.
For example,

```sql
SET SCHEMA MovieSchema;
```

### 2.3 Catalogs

```sql
SET CATALOG <catlog name>
```

### 2.4 Clients and Servers in SQL Environment

A SQL environment is more than a collection of catalogs and schemas.
It contains elements whose purpose is to support operations on
the database or databases represented by those catalogs and schemas.

### 2.5 Connections

If we wish to run some program involving SQL at a host where a SQL
client exist, then we may open a connection between the client and server
by executing a SQL statement.

```sql
CONNECT TO <server name> AS <connection name>
  AUTHORIZATION <name and password>
```

The connection name can be used to refer to the connection later on.
The reason we might have to refer to the connection is that SQL
allows several connections to be opened by the user, but only can
be active at any time. To switch among connections, we can make `conn1`
become the active connection by the statement:

```sql
SET CONNECTION conn1;
```

Whatever connection was currently active becomes *dormant* until
it is reactivated with another `SET CONNECTION` statement that
mentions it explicitly.

We also use the name when we drop the connection. We can drop
connection `conn1` by

```sql
DISCONNECT conn1;
```

### 2.6 Sessions

The SQL operations that are performed while a connection is active form
a *session*. The session lasts as long as the connection that created it.

Each session has a current catalog and a current schema within the catalog.

### 2.7 Modules

A *module* is the SQL term for an application program. The SQL standard
suggests that there are three kinds of modules, but insists only
that a SQL implementation offer the user at least one of these types.

+ *Generic SQL Interface*.
+ *Embedded SQL*
+ *True Modules*

## 3. Thee SQL/Host-Language Interface

To this point, we have used the *generic SQL interfaces*. In real systems, there
is a program in some conventional *host* language such as C, but some
of the steps in this program are actually SQL statements.

There are two ways this embedding could take place.

+ *Call-Level Interface*.
+ *Directly Embedded SQL*.

I omit detail.
