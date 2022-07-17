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

Like STL, Pintos defines `list_push_front` and `list_push_back`.

```c
void list_push_front(struct list* list, struct list_elem* elem) {
  list_insert(list_begin(list), elem);
}

void list_push_back(struct list* list, struct list_elem* elem) {
  list_insert(list_end(list), elem);
}
```

Now that we have defined the insert operation, now we will get to remove operation.

```c
struct list_elem* list_remove(struct list_elem* elem) {
  ASSERT(is_interior(elem));
  elem->prev->next = elem->next;
  elem->next->prev = elem->prev;
  return elem->next;
}
```

And there are some basic operations, I omit detail here.

```c
struct list_elem* list_pop_front(struct list* list) {
  struct list_elem* front = list_front(list);
  list_remove(front);
  return front;
}

struct list_elem* list_pop_back(struct list* list) {
  struct list_elem* back = list_back(list);
  list_remove(back);
  return back;
}
```

And Pintos defines `list_front` and `list_back` to get the first
element and the last element.

```c
struct list_elem* list_front(struct list* list) {
  ASSERT(!list_empty(list));
  return list->head.next;
}

struct list_elem* list_back(struct list* list) {
  ASSERT(!list_empty(list));
  return list->tail.prev;
}
```

The interesting function Pintos defines is `list_splice` which
removes elements `[FIRST, LAST]` and inserts them just before `BEFORE`.

```c
void list_splice(struct list_elem* before, struct list_elem* first, struct list_elem* last) {
  ASSERT(is_interior(before) || is_tail(before));
  if (first == last)
    return;
  last = list_prev(last);

  ASSERT(is_interior(first));
  ASSERT(is_interior(last));

  first->prev->next = last->next;
  last->next->prev = first->prev;

  /* Splice FIRST...LAST into new list. */
  first->prev = before->prev;
  last->next = before;
  before->prev->next = first;
  before->prev = last;
}
```

Because of the node is just simply allocated on the stack, so we just
drop the node with two lines `first->prev->next = last->next`
and `last->next->prev = first->prev`.

Next, Pintos defines `list_size` to get the number of elements in the list.

```c
size_t list_size(struct list* list) {
  struct list_elem* e;
  size_t cnt = 0;
  for(e = list_begin(list); e != list_end(list); e = list_next(e))
    cnt++;
  return cnt;
}
```

And `list_empty` returns true if list is empty.

```c
bool list_empty(struct list* list) {
  return list_begin(list) == list.end(list);
}
```

Pintos defines `swap` to swap the node itself, actually this is a
good idea to do some algorithm questions.

```c
static void swap(struct list_elem** a, struct list_elem** b) {
  struct list_elem* t = *a;
  *a = *b;
  *b = t;
}
```

And Pintos defines `is_sorted` to determine interval `[A,B]` are in order
according to `LESS`.

```c
static bool is_sorted(struct list_elem* a, struct list_elem* b, list_less_func* less, void* aux) {
  if(a != b)
    while((a = list_next(a)) != b)
      if(less(a, list_prev(a), aux))
        return false;
  return true;
}
```
