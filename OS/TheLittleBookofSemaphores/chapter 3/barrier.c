#include <stdio.h>
#include <pthread.h>

/*
  * In this implementation of barrier for C code,
  * We could use `pthread_cond_broadcast` or like
  * the pattern in Python code. However, if we decide
  * not to use semaphore, we won't have the problem
  * describe in the Python code, because we does a
  * more `pthread_cond_signal` which won't affect
  * the behavior. However, semaphore has a state, so
  * this is the problem. So we could simply just make
  * the condition become `count % THREAD_NUM == 0` to
  * make a useable-barrier.
*/

#define THREAD_NUM 4

pthread_mutex_t lock;
pthread_cond_t cond;

int count = 0;

void* barrier(void* arg) {
  printf("Hello Barrier Example\n");

  pthread_mutex_lock(&lock);
  count++;
  pthread_mutex_unlock(&lock);

  pthread_mutex_lock(&lock);
  if(count % THREAD_NUM == 0) {
    pthread_cond_signal(&cond);
  } else {
    pthread_cond_wait(&cond, &lock);
    pthread_cond_signal(&cond);
  }

  pthread_mutex_unlock(&lock);
  printf("Done\n");
}

int main() {
  pthread_mutex_init(&lock, NULL);
  pthread_cond_init(&cond, NULL);

  pthread_t thread[THREAD_NUM];

  for(int i = 0; i < THREAD_NUM; ++i) {
    pthread_create(&thread[i], NULL, barrier, NULL);
  }
  for(int i = 0; i < THREAD_NUM; ++i) {
    pthread_join(thread[i], NULL);
  }

  // We could do a more round.
  for(int i = 0; i < THREAD_NUM; ++i) {
    pthread_create(&thread[i], NULL, barrier, NULL);
  }
  for(int i = 0; i < THREAD_NUM; ++i) {
    pthread_join(thread[i], NULL);
  }

  return 0;
}