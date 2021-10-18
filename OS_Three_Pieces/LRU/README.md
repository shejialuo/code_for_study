# LRU

The idea is so simple. It holds a vector. When a new page
is coming:

+ Tt first find whether there is an existing page in the vector.
  + If so, let `index = i`.
  + Else, let `index = length - 1`
+ Then it goes from index to 1.

```c
for(int i = index; i > 0; --i) {
  vector[i] = vector[i - 1];
}
```
