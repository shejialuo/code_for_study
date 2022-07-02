#include <stdio.h>
#include <pthread.h>

pthread_mutex_t a_arrived;
pthread_cond_t a_arrived_cond;
pthread_mutex_t b_arrived;
pthread_cond_t b_arrived_cond;

int a = 0;
int b = 0;

void* print_for_thread_A(void* arg) {
  printf("statement a1\n");
  a = 1;
  pthread_cond_signal(&a_arrived_cond);
  pthread_mutex_lock(&b_arrived);
  while (b == 0)
    pthread_cond_wait(&b_arrived_cond, &b_arrived);
  pthread_mutex_unlock(&b_arrived);
  printf("statement a2\n");
}

void* print_for_thread_B(void* arg) {
  printf("statement b1\n");
  b = 1;
  pthread_cond_signal(&b_arrived_cond);
  pthread_mutex_lock(&a_arrived);
  while (a == 0)
    pthread_cond_wait(&a_arrived_cond, &a_arrived);
  pthread_mutex_unlock(&a_arrived);
  printf("statement b2\n");
}

int main() {
  pthread_mutex_init(&a_arrived, NULL);
  pthread_mutex_init(&b_arrived, NULL);
  pthread_cond_init(&a_arrived_cond, NULL);
  pthread_cond_init(&b_arrived_cond, NULL);

  pthread_t A;
  pthread_t B;

  pthread_create(&A, NULL, print_for_thread_A, NULL);
  pthread_create(&B, NULL, print_for_thread_B, NULL);

  pthread_join(A, NULL);
  pthread_join(B, NULL);

  return 0;
}
