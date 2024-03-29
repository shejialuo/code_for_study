# Project 1 Buffer Pool

## Extensible Hash Table

The first idea is to understand how extensible hash table works. Actually,
it is important to understand the functionality of the local depth and the
global depth.

The global depth is aims at distrusting the buckets at a high level. However,
there is a local depth, which could be less than global depth. When this
situation happens, there must be two different directories pointing to
the same bucket.

So there could be two situations. It is a wonderful design.

We assume that the local depth is `j` and the global depth is `i`. When inserting
to a fulled bucket:

1. If `j < i`, the nothing needs to be done to the bucket array. We
    1. Split the block `B` into two.
    2. Distribute records in `B` to the two blocks, based on the value of their
    `j+1`st bit, 0 stay in `B` and those with 1 there go to the new block.
    3. Plus the one to the local depth.
    4. Adjust the pointers in the bucket array so entries that formerly pointed
    to `B` now point either to `B` or the new block, depending on their `j+1`st bit.
2. However, there may be cases where all the records of `B` may go into one of the
two blocks into which it was split. If so, we need to repeat the process the process
until we have successfully inserted into the hash table.
3. If `i==j`, the must plus 1 to the global depth. We double the length of the bucket
array. Suppose $w$ is a sequence of `i` bits indexing one of the entries in the previous
bucket array. In the new bucket array, the entries index by `0w` and `1w` each point
to the same block that the `w` entry used to point to. That is, the two new entries
share the block, and the block itself does not change. And go back to 1.

## LRU-K

Well, it is important to understand how to implement a LRU-1 first. We could easily do
this to use two data structures:

+ `list<T> items`
+ `unordered_map<T, typename <list<T>>::iterator> tables`

When we insert a new value, if it is in the `tables`. We should move it to the head of
the list and update the tables.

It is just the same. And we could use the following structures:

+ `list<T> items_infinity`
+ `unordered_map<T, typename <list<T>>::iterator> tables_infinity`
+ `list<T> items_ready`
+ `unordered_map<T, typename <list<T>>::iterator> tables_ready`

When the access time is less than the `k`, we put the itm in `items_infinity` and
everything else is the same as LRU. If one of item in the `items_infinity` access
time exceeds the `k`, we move it to the head of the `tables_ready`.

When we want to get a page, we could easily first finds in the `items_infinity` and
secondly find in the `items_ready`. What a good abstraction.
