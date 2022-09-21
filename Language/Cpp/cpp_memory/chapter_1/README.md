# Chapter 1 Object Lessons

In C, a data abstraction and the operations that perform on it are
declared separately. For example, if we declare a struct `Point3d`, such
as the following:

```c
typedef struct point3d {
  float x;
  float y;
  float z;
}Point3d;
```

The operation to print a particular `Point3d` might be defined
either as a function or for efficiency as a preprocessor macro.

In C++, `Point3d` is likely to be implemented as an independent
abstract data type (ADT):

```c++
class Point3d {
public:
  Point3d(float x = 0.0, float y = 0.0, float z = 0.0)
    : _x(x), _y(y), _z(z) {}
  float x() { return _x; }
  float y() { return _y; }
  float z() { return _z;}

  void x(float xval) { _x = xval; }
  // ...etc...
private:
  float _x;
  float _y;
  float _z;
}

inline ostream&
operator<<( ostream &os, const Point3d &pt ) {
 os << "( " << pt.x() << ", "
 << pt.y() << ", " << pt.z() << " )";
};
```

Also you can implement as a two- or three-level class hierarchy.

These are obviously not only very different styles of programming,
but also very different ways of thinking about our programs.

There are no additional layout costs for supporting the class `Point3d`. The
three coordinate data members are directly contained within each class object,
as they are in the C struct. The member functions, although included in
the class declaration, are not reflected in the object layout; one
copy only of each non-inline member function is generated.
Each inline function has either zero or one definition of
itself generated within each module in which it is used. The `Point3d`
class has no space or runtime penalty in supporting encapsulation.
As you will see, the primary layout and access-time overheads within C++
are associated with the *virtual*, that is,

+ the virtual function mechanism in its support of an efficient run-time binding, and
+ a virtual base class in its support of a single, shared instance
of a base class occurring multiple times within an inheritance hierarchy.

## 1.1 The C++ Object Model

In C++, there are two flavors of class data members: static and non-static.
and three flavors of class member functions: static, non-static, and virtual.
Given the following declaration of a class `Point`:

```c++
class Point {
public:
  Point(float xval);
  virtual ~Point();
  float x() const;
  static int PointCount();
protected:
  virtual ostream& print( ostream &os ) const;
  float _x;
  static int _point_count;
}
```

How is the class `Point` to be represented within the machine?

### 1.1.1 A Simple Object Model

Our first object model is admittedly very simple. It might be
used for a C++ implementation designed to minimize the complexity
of the compiler at the expense of space and runtime efficiency.
In this simple model, an object is a sequence of slots, where each slot
points to a member.
The members are assigned a slot in the order of their declarations.

![A simple object model](https://s2.loli.net/2022/08/09/R4oa5f7gCUE2mDu.png)

In this simple model, the members themselves are not placed within the
object. Only pointers addressing the members are placed within
the object. Doing this avoids problems from member's being quite
different types and requiring different amounts storage.

Although this model is not used in practice, this simple concept
of an index or slot number is the one that has been developed into
the C++ pointer-to-member concept.

### 1.1.2 A Table-driven Object Model

For an implementation to maintain a uniform representation for
the objects of all classes, an alternative object model might factor
out all member specific information, placing it in a data member and member
function pair of table. The class object contains the pointers
to the two member tables. The member function table is a sequence
of slots, with each slot address a member. The data member directly
holds the data. This is shown below.

![A table-driven object model](https://s2.loli.net/2022/08/09/PVQAXiRCq4FxjHO.png)

Although this model is not used in practice within C++, the concept of
a member function table has been the traditional implementation
supporting efficient runtime resolution of virtual functions.

### 1.1.3 A Table-driven Object Model

C++ object model is derived from the simple object model by optimizing
for space and access time. Non-static data members are allocated
directly within each class object. Static data members are stored
outside the individual class object. Static and non-static
function members are also hoisted outside the class object.
Virtual functions are supported in two steps:

+ A table of pointers to virtual functions is generated for each class.
+ A single pointer to the associated virtual table is inserted
within each class object(traditionally, this has been called the `vptr`).
The setting, resetting, and not setting of the `vptr` is handled
automatically through code generated within each class constructor,
destructor, and copy assignment operator. The `type_info` object
associated with each class in support of runtime type identification (RTTI)
is also addressed within the virtual table, usually within the table's first slot.

Below figure illustrates the general C++ Object Model for `Point` class.

![C++ object model](https://s2.loli.net/2022/08/10/q3ElRSimb9UHWGJ.png)

#### Adding Inheritance

C++ supports both *single* inheritance and *multiple* inheritance. Moreover,
the inheritance may be specified as *virtual*. In the case of
virtual inheritance, only a single occurrence of the base class is maintained
regardless of how many times the class is derived from within the
inheritance chain.

In a simple base class object model, each base class might be assigned
to a slot within the derived class object. Each slot holds the address
of the base class object. The primary draw back to this scheme
is the space and access-time overhead of the indirection. A
benefit is that the size of the class object is unaffected by
changes in the size of its associated base classes.

Alternatively, we can use a base table model. Here, a base class
table is generated for which each slot contains the address of an
associated base class. Each class object contains a *bptr* initialized to
address its base class table. The primary drawback to this strategy,
is both the space and access-time overhead of the indirection.
One benefit is a uniform representation of inheritance within
each class object. A second benefit would be the ability to grow, shrink,
or otherwise modify the base class table without changing the size
of the class objects.

The original inheritance model supported by C++ forgoes all indirection;
the data members of the base class are directly stored within the
derived class object. This offers the most compact and most efficient
access of the base class members.

The introduction of virtual base classes into the language
required some form of indirect base class representation.
The original model of virtual base class support added a pointer
into the class object for each associated virtual base class.

#### How the Object Model Effects Programs

In practice, what does this mean for the programmer? Support for
the object model results in both modifications of the existing program
code and the insertion of additional code. For example.

```c++
X foobar() {
  X xx;
  X *px = new X;
  // foo() is a virtual function
  xx.foo();
  px->foo();
  delete px;
  return xx;
}
```

The likely internal transformation of the function looks as follows:

```c++
void foobar(x &_result) {
  / _result replaces local xx...
  _result.X::X();
  px = _new(sizeof(X));
  if(px != 0)
    px->X::X();

  foo(&_result);
  (*px->_vtbl[ 2 ])(px)
  if(px != 0) {
    (*px->_vtbl[ 1 ])(px);
    _delete(px);
  }
  return;
}
```
