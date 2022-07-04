#include <pthread.h>
#include <stdio.h>

typedef struct __node_t {
  int value;
  struct __node_t *next;
}node_t;

typedef struct __queue_t {
  node_t *head;
  node_t *tail;
  pthread_mutex_t headLock;
  pthread_mutex_t tailLock;
}queue_t;

void queue_init(queue_t* q) {
  node_t *sentinel = malloc(sizeof(node_t));
  sentinel->next = NULL;
  q->head = q->tail = sentinel;
  pthread_mutex_init(&q->headLock, NULL);
  pthread_mutex_init(&q->tailLock, NULL);
}

void queue_enqueue(queue_t *q, int value) {
  node_t *node = malloc(sizeof(node_t));
  node->value = value;
  node->next = NULL;

  pthread_mutex_lock(&q->tailLock);
  q->tail->next = node;
  q->tail = node;
  pthread_mutex_unlock(&q->tailLock);
}

int queue_dequeue(queue_t *q, int *value) {
  pthread_mutex_lock(&q->headLock);
  node_t *node = q->head;
  node_t *newHead = node->next;
  if(newHead == NULL) {
    pthread_mutex_unlock(&q->headLock);
    return -1;
  }
  *value = newHead->value;
  q->head = newHead;
  pthread_mutex_unlock(&q->headLock);
  free(node);
  return 0;
}
