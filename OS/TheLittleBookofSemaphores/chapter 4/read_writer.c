#include <pthread.h>
#include <stdio.h>

#define BUF_SIZE 10

int buf[BUF_SIZE] = {2};
int count = 0;
int reader_num = 0;

struct __write{
  int index;
  int value;
};

// Auxiliary function
void put(struct __write* w) {
  if(w->index >= BUF_SIZE) return;
  buf[w->index] = w->value;
}

// Auxiliary function
int get(int index) {
  if(index < BUF_SIZE)
    return buf[index];
  return -1;
}

pthread_mutex_t mutex;
pthread_cond_t room_empty;

void* writer(void* arg) {
  pthread_mutex_lock(&mutex);
  while(reader_num != 0)
    pthread_cond_wait(&room_empty, &mutex);
  put((struct __write*)arg);
  pthread_mutex_unlock(&mutex);
}

void* reader(void* arg) {
  pthread_mutex_lock(&mutex);
  reader_num++;
  printf("%d\n", get(*(int*)arg));
  reader_num--;
  if(reader_num == 0)
    pthread_cond_broadcast(&room_empty);
  pthread_mutex_unlock(&mutex);
}


int main() {

  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&room_empty, NULL);
  pthread_t w[100];
  pthread_t r[100];

  struct __write write_info = {0, 0};

  for(int i = 0; i < 100; ++i) {
    int index = i % BUF_SIZE;
    write_info.index = index;
    write_info.value = i;
    pthread_create(&w[i], NULL, writer, &write_info);
    pthread_create(&r[i], NULL, reader, &index);
  }

  for(int i = 0; i < 100; ++i) {
    pthread_join(w[i], NULL);
    pthread_join(r[i], NULL);
  }

  return 0;
}
