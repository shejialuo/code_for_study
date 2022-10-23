# STL Iterators

## Iterator Categories

Anything that behaves like an iterator is an iterator.

### Output Iterators

Output iterators can only step forward with write access. Thus, you can
assign new value only element-by-element. You can't use an output iterator
to iterate twice over the same range.

```c++
while (...) {
  *pos = ...;
  ++pos;
}
```

| Expression | Effect |
|:---:|:---:|
| *iter = val | Writes val to where the iterator refers |
| ++iter | Steps forward (returns new position) |
| iter++ | Steps forward (returns old position) |
| TYPE(iter) | Copies iterator (copy constructor) |

Often, iterators can read and write values. For this reason, all reading
iterators might have the additional ability to write. In that case,
they are called *mutable* iterators.

All `const_iterator`s provided by containers and their member functions
`cbegin()` and `cend()` are not output iterators.

### Input Iterators

Input iterators can only step forward element-by-element with read access. Thus,
they return values element-wise.

|   Expression   |                         Effect                         |
|:--------------:|:------------------------------------------------------:|
|      *iter     |       Provides read access to the actual element       |
|  iter->member  | Provides read access to a member of the actual element |
|     ++iter     |          Steps forward (returns new position)          |
|     iter++     |           Steps forward (return old position)          |
| iter1 == iter2 |         Returns whether two iterators are equal        |
|   TYPE(iter)   |           Copies iterator (copy constructor)           |

Input iterators can read elements only once. Thus, if you copy an input iterator
and let the original and the copy read forward, they might iterate over different
values.

For input iterators, opreators `==` and `!=` are provided only to check whether
an iterator is equal to a `past-the-end iterator`. This is required because
operations that deal with input iterators usually do the following:

```c++
while (pos != end) {
  ++pos;
}
```

### Forward Iterators

Forward iterators are input iterators that provide additional guarantees while reading
forward.

|   Expression   |                         Effect                         |
|:--------------:|:------------------------------------------------------:|
|      *iter     |       Provides read access to the actual element       |
|  iter->member  | Provides read access to a member of the actual element |
|     ++iter     |          Steps forward (returns new position)          |
|     iter++     |           Steps forward (return old position)          |
| iter1 == iter2 |         Returns whether two iterators are equal        |
| iter1 != iter2 |       Returns whether two iterators are not equal      |
|     TYPE()     |         Creates iterator (default constructor)         |
|   TYPE(iter)   |           Copies iterator (copy constructor)           |
|  iter1 = iter2 |                   Assigns an iterator                  |

Unlike for input iterators, it is guaranteed that for two forward iterators that
refer to the same element, operator `==` yields `true` and that they will refer
to the same value after both are incremented.

Forward iterators are provided by the following objects and types:

+ Class `forward_list<>`.
+ Unordered containers.

### Bidirectional Iterators

Bidirectional Iterators are forward iterators that provide the additional ability
to iterate backward over the elements. Thus, they provide the decrement operator
to step backward.

| Expression |                 Effect                |
|:----------:|:-------------------------------------:|
|   --iter   | Steps backward (returns new position) |
|   iter--   | Steps backward (returns old position) |

Bidirectional iterators are provided by the following objects and types:

+ Class `list<>`
+ Associative containers.

### Random-Access Iterators

Random-access iterators provide all the abilities of bidirectional iterators plus random
access. Thus, they provide operators for *iterator arithmetic*. That is, they can add
and subtract offsets, process differences, and compare iterators with relational
operators, such as `<` and `>`.

|  Expression  |                          Effect                          |
|:------------:|:--------------------------------------------------------:|
|    iter[n]   |      Provides access to the element that has index n     |
|    iter+=n   | Steps n elements forward (or backward, if n is negative) |
|    iter-=n   | Steps n elements backward (or forward, if n is negative) |
|    iter+n    |       Returns the iterator of the nth next element       |
|    n+iter    |       Returns the iterator of the nth next element       |
|    iter-n    |     Returns the iterator of the nth previous element     |
|  iter1-iter2 |       Returns the distance between iter1 and iter2       |
|  iter1<iter2 |           Returns whether iter1 is before iter2          |
|  iter1>iter2 |           Returns whether iter1 is after iter2           |
| iter1<=iter2 |         Returns whether iter1 is not after iter2         |
| iter1>=iter2 |         Returns whether iter1 is not before iter2        |

Random-access iterators are provided by the following objects and types:

+ Containers with random access
+ Strings
+ Ordinary C-style arrays

## Auxiliary Iterator Functions

+ `advance`
+ `next`
+ `prev`
+ `distance`
+ `iter_swap`: swap the values to which two iterators refer.

## Iterator Adapters

### Reverse Iterators

*Reverse iterators* redefine increment and decrement operators so that they behave
in reverse. Thus, if you use these iterators instead of ordinary iterators, algorithms
process elements in reverse order.

+ `rbegin()` returns the position of the first element of a reverse iteration. Thus, it
returns the position of the last element.
+ `rend()` returns the position after the last element of a reverse iteration. Thus, it returns
the position *before* the first element.

You can convert normal iterators into reverse iterators. Naturally, the iterators must be
bidirectional iterators, but note the logical position of an iterator is moved during
the conversion.

[reverseIterator2.cpp](./reverseIterator2.cpp)

If you print the value of an iterator and convert the iterator into a reverse iterator, the
value has changed. This is not a bug; it's a feature! This behavior is a consequence of the
fact that ranges are half open.

You can convert reverse iterators back into normal iterators using `base()`.

### Insert Iterators

An insert iterator transforms an assignment into an insertion. However, two operations
are involved: operator `*` returns the current element of the iterator; second, operator `=`
assigns a new value. Implementations of insert iterators usually use the following
two-trick:

+ Operator `*` is implemented as a no-op that simple returns `*this`.
+ The insert iterator calls the `push_back()`, `push_front()` or `insert()` member function
of the container.

|  Expression  |        Effect       |
|:------------:|:-------------------:|
|     *iter    | No-op(returns iter) |
| iter = value |     Insert value    |
|    ++iter    | No-op(returns iter) |
|    iter++    | No-op(returns iter) |

The C++ standard library provides three kinds of insert iterator: back inserters, front
inserters, and general inserters.

|       Name       |          Class         |  Called Function  |       Creation       |
|:----------------:|:----------------------:|:-----------------:|:--------------------:|
|   Back inserter  | back_inserter_iterator |  push_back(value) |  back_inserter(coll) |
|  Front inserter  |  front_insert_iterator | push_front(value) | front_inserter(coll) |
| General inserter |     insert_iterator    | insert(pos,value) |  inserter(cont,pos)  |

### Stream Iterators

A *stream iterator* is an iterator adapter that allows you to use a stream as a source or
destination of algorithms.

*Ostream iterators* write assigned value to an output stream. By using ostream iterators,
algorithms can write directly to streams. The implementation of an ostream iterator uses
the same concept as the implementation of insert iterators. The only difference is
that they transform the assignment of a new value into an output operation by using
operator `<<`.

|              Expression              |                       Effect                       |
|:------------------------------------:|:--------------------------------------------------:|
|     ostream_iterator<T>(ostream)     |       Creates an ostream iterator for ostream      |
| ostream_iterator <T>(ostream, delim) | Creates an ostream iterator for ostream with delim |
|                 *iter                |                No-op (returns iter)                |
|             iter = value             |          Writes value using ostream<<value         |
|                ++iter                |                No-op (returns iter)                |
|                iter++                |                No-op (returns iter)                |

*Istream Iterators* are the counterparts of ostream iterators. An istream iterator reads elements
from an input stream.

### Move Iterators

Since C++11, an iterator adapter is provided that converts any access to the underlying
element into a move operation.

```c++
 move operation. For example:
std::list<std::string> s;
...
std::vector<string> v1(s.begin(), s.end());
std::vector<string> v2(make_move_iterator(s.begin()),make_move_iterator(s.end()));
```
