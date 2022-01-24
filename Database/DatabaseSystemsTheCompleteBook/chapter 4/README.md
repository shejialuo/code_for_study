# Chapter 4 High-Level Database Models

Let us consider the process whereby a new database, such as our movie database,
is created. The below figure suggests the process. We begin with a design phase.

![The database modeling and implementation process](https://i.loli.net/2021/10/04/LFCOQwxeftvXADm.png)

In practice it is often easier to start with a higher-level model and convert the
design to the relational model.

## 1. The Entity/Relationship Model

In the *entity-relationship model* (or E/R model), the structure of data is
represented graphically, using three principal element types:

+ Entity sets.
+ Attributes.
+ Relationships.

### 1.1 Entity sets

An *entity* is an abstract object of some sort, and a collection of similar entities
forms an *entity set*.

### 1.2 Attributes

Entity sets have associated *attributes*, which are properties of the entities
in that set.

In our version of the E/R model, we shall assume that attributes are of primitive
types.

### 1.3 Relationships

*Relationships* are connections among two or more entity sets.

### 1.4 Entity-Relationship Diagrams

An E/R *diagram* is a graph representing entity sets, attributes, and relationships.
Elements of each these kinds are represented by nodes of the graph, and we
use a special shape of node to indicate the kind, as follows:

+ Entity sets are represented by rectangles.
+ Attributes are represented by ovals.
+ Relationships are represented by diamonds.

Edges connect an entity set to its attributes and also connect a relationship to
its entity sets.

The following figure represents a simple database about movies.

![An entity-relationship diagram for the movie database]()

### 1.5 Instances of an E/R Diagram

For each entity set, the database instance will have a particular finite set of
entities. Each of these entities has particular values for each attribute.
A relationship $R$ that connects $n$ entity sets $E_{1},E{2},\dots,E_{n}$ may be
imagined to have an "instance" that consists tuples $(e_{1},e_{2},\dots,e_{n})$,
where $e_{i}$ is chosen from the entities that are in the current instance of
entity set $E_{i}$.

For example, an instance of the `Stars-in` relationship could be visualized as a
table with pairs such as:

<!-- TODO: add the picture -->
![An instance of the Stars-in relationship](.)

### 1.6 Multiplicity of Binary E/R Relationships

In general, a binary relationship can connect any number of one of its entity sets
to any number of members of the other entity set. However, it is common for there
to be a restriction on the "multiplicity" of a relationship. Suppose $R$ is a
relationship connecting entity sets $E$ and $F$. Then:

+ If each member of $E$ can be connected by $R$ to at most one member of $F$, then
we say that $R$ is *many-one* from $E$ to $F$.
+ If $R$ is both many-one from $E$ to $F$ and many-one from $F$ to $E$, then we say
that $R$ is *one-one*.
+ If $R$ is neither many-one from $E$ to $F$ or from $F$ to $E$, then we say $R$
is *many-many*.

### 1.7 Multiway Relationships

In practice, ternary or high-degree relationships are rare, but they occasionally
are necessary to reflect the true state of affairs.

### 1.8 Roles in Relationships
