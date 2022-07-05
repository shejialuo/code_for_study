#include <pthread.h>
#include <stdio.h>

#define BUF_SIZE 10
#define LOOP 1000000

int buf[BUF_SIZE];
int count = 0;

// Auxiliary function
void put(int value) {
  buf[count] = value;
  count++;
}

// Auxiliary function
int get() {
  int value = buf[count - 1];
  count--;
  return value;
}

pthread_mutex_t mutex;

pthread_cond_t full;
pthread_cond_t empty;

void* producer(void* arg) {
  for(int i = 0; i < LOOP; ++i) {
    pthread_mutex_lock(&mutex);
    while(count == BUF_SIZE) {
      pthread_cond_wait(&empty, &mutex);
    }
    put(i);
    pthread_cond_signal(&full);
    pthread_mutex_unlock(&mutex);
  }
}

void* consumer(void* arg) {
  for(int i = 0; i < LOOP; ++i) {
    pthread_mutex_lock(&mutex);
    while(count == 0) {
      pthread_cond_wait(&full, &mutex);
    }
    printf("%d\n", get());
    pthread_cond_signal(&empty);
    pthread_mutex_unlock(&mutex);
  }
}


int main() {

  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&full, NULL);
  pthread_cond_init(&empty, NULL);

  pthread_t p[3];
  pthread_t c[3];

  for(int i = 0; i < 3; ++i) {
    pthread_create(&p[i], NULL, producer, NULL);
    pthread_create(&c[i], NULL, consumer, NULL);
  }

  for(int i = 0; i < 3; ++i) {
    pthread_join(p[i], NULL);
    pthread_join(c[i], NULL);
  }

  return 0;
}
