# Chapter 11 Hash Tables

## 11.1 Direct-address tables

Direct addressing is a simple technique that works well when the universe $U$
of keys is reasonably small. Suppose that an application needs a dynamic set
in which each element has a key drawn from the universe $U = {0,1,\dots, m -1}$,
where $m$ is not too large. We shall assume that not two elements have the
same key.

To represent the dynamic set, we use an array, or *direct-address table*, denoted
by `T[0..m-1]`, in which each position, or *slot*, corresponds to a key in the
universe $U$.

## 11.2 Hash tables

The downside of direct addressing is obvious: if the universe $U$ is large, storing
a table $T$ of size $|U|$ may be impractical, or even impossible.

With direct addressing, an element with key $k$ is stored in slot $k$. With hashing,
this element is stored in slot $h(k)$; that is, we use a *hash function* to compute
the slot from the key $k$.

There is one hitch: two keys may hash to the same slot. We call this situation a *collision*.

### Collision resolution by chaining

In *chaining*, we place all the elements that hash to the same slot into the same linked list.
Given a hash table $T$ with $m$ slots that stores $n$ elements, we define the *load factor*
$\alpha$ for $T$ as $n / m$.

The unsuccessful search takes average-case time $\Theta(1 + \alpha)$, under the assumption
of simple uniform hashing. The successful search takes average-case time $\Theta(1 + \alpha)$, under the assumption of simple uniform hashing.

## 11.3 Open addressing

In *open addressing*, all elements occupy the hash table itself. That is, each table entry
contains either an element of the dynamic set of `NULL`.
