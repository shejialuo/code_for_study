# List Data Structure

Pintos defines list data structure at `src/lib/kernel/list.c`. You should
first carefully read the comment of the file and the corresponding header file.
It's clear to clarify the most important concepts.

First, we look at the data structure.

```c
struct list_elem {
  struct list_elem *prev;
  struct list_elem *next;
}

struct list {
  struct list_elem head;
  struct list_elem tail;
}
```

Pintos uses `list_init` to initialize the linked list.

```c
void list_init(struct list* list) {
  ASSERT(list != NULL);
  list->head.prev = NULL;
  list->head.next = &list->tail;
  list->tail.prev = &list->head;
  list->tail.next = NULL;
}
```

First, Pintos defines some auxiliary functions, it is easy to understand.

```c
static bool is_sorted(struct list_elem* a, struct list_elem* b, list_less_func* less,
                      void* aux) UNUSED;

static inline bool is_head(struct list_elem* elem) {
  return elem != NULL && elem->prev == NULL && elem->next != NULL;
}

static inline bool is_interior(struct list_elem* elem) {
  return elem != NULL && elem->prev != NULL && elem->next != NULL;
}

static inline bool is_tail(struct list_elem* elem) {
  return elem != NULL && elem->prev != NULL && elem->next == NULL;
}
```

Pintos uses `list_begin` to return the first element of the list.
Also, Pintos uses `list_rbegin` to return the last element of the list.

```c
struct elem* list_begin(struct list* list) {
  ASSERT(list != NULL);
  return list->head.next;
}

struct elem* list_rbegin(struct list* list) {
  ASSERT(list != NULL);
  return list->tail.prev;
}
```

If you have studied the STL, you should be familiar with this interface.
Actually, Pintos uses the nearly same interface as c++ STL.

Let's look at `list_end` and `list_rend`.

```c
struct list_elem* list_end(struct list* list) {
  ASSERT(list != NULL);
  return &list->tail;
}

struct list_elem* list_rend(struct list* list) {
  ASSERT(list != NULL);
  return &list->head;
}
```

And Pintos encapsulates the code to get the `head` node and `tail` node.

```c
struct list_elem* list_head(struct list* list) {
  ASSERT(list != NULL);
  return &list->head;
}

struct list_elem* list_tail(struct list* list) {
  ASSERT(list != NULL);
  return &list->tail;
}
```

The interesting part is the `list_next` function, like STL return the
next iterator. The same is for `list_prev` function.

```c
struct list_elem* list_next(struct list_elem* elem) {
  ASSERT(is_head(elem) || is_interior(elem));
  return elem->next;
}

struct list_elem* list_prev(struct list_elem* elem) {
  ASSERT(is_interior(elem) || is_tail(elem));
  return elem->prev;
}
```

Now comes interesting part. Pintos defines `list_insert` to insert
elem before `before`.

```c
void list_insert(struct list_elem* before, struct list_elem* elem) {
  ASSERT(is_interior(before) || is_tail(before));
  ASSERT(elem != NULL);

  elem->prev = before->prev;
  elem->next = before;
  before->prev->next = elem;
  before->prev = elem;
}
```
