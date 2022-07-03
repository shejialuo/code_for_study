#include <stdio.h>
#include <pthread.h>

pthread_mutex_t lock;

int count = 0;

void *count_thread_A(void* arg) {
  pthread_mutex_lock(&lock);
  count++;
  pthread_mutex_unlock(&lock);
}

void *count_thread_B(void* arg) {
  pthread_mutex_lock(&lock);
  count++;
  pthread_mutex_unlock(&lock);
}

int main() {
  pthread_mutex_init(&lock, NULL);
  pthread_t A, B;
  pthread_create(&A, NULL, count_thread_A, NULL);
  pthread_create(&B, NULL, count_thread_B, NULL);
  pthread_join(A, NULL);
  pthread_join(B,NULL);
  printf("%d\n", count);
  return 0;
}