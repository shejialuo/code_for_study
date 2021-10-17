#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "mapreduce.h"

#define num_partition 10

typedef struct partition partition_t;
typedef struct key_value key_value_t;
typedef struct list list_t;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int done = 0;
int word_count = 0;

Mapper mapper_global;

Reducer reducer_global;

struct list {
  char *value;
  struct list *next;
};

struct key_value {
  char *key;
  pthread_mutex_t mutex;
  list_t *list;
  struct key_value *next;
};

struct partition {
  pthread_mutex_t mutex ;
  key_value_t *keyValue ;
};

partition_t partitionPage[num_partition];

unsigned long MR_DefaultHashPartition(char *key, int num_partitions) {
  unsigned long hash = 5381;
  int c;
  while ((c = *key++) != '\0')
    hash = hash * 33 + c;
  return hash % num_partitions;
}

void insertValue(partition_t *p, char *key, char *value) {
  key_value_t *iter = p->keyValue;
  while(iter != NULL) {
    if (strcmp(iter->key, key) == 0)
      break;
    iter = iter->next;
  }

  /* 
    * When key_value list is empty or can't find the corresponding key
    * Insert a new node.
  */
  if (iter == NULL) {
    pthread_mutex_lock(&p->mutex);

    key_value_t *newKeyValueList = (key_value_t *)malloc(sizeof(key_value_t));
    
    newKeyValueList->next = p->keyValue;
    p->keyValue = newKeyValueList;

    pthread_mutex_init(&newKeyValueList->mutex, NULL); 

    newKeyValueList->key = (char *)malloc(strlen(key) + 1);
    strcpy(newKeyValueList->key, key);

    newKeyValueList->list = NULL;

    iter = newKeyValueList;

    pthread_mutex_unlock(&p->mutex);
  }

  pthread_mutex_lock(&iter->mutex);
  list_t *newValueList = (list_t *)malloc(sizeof(list_t));

  newValueList->next = iter->list;
  iter->list = newValueList;

  newValueList->value = (char *)malloc(strlen(value) + 1);
  strcpy(newValueList->value, value);
  pthread_mutex_unlock(&iter->mutex);
}

void *start_mapper(void *arg) {
  char *filename = (char *)arg;
  
  mapper_global(filename);

  pthread_mutex_lock(&mutex);
  done = 1;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);

  return (void *)0;
}

void MR_Emit(char *key, char *value) {
  // int partitionPageID = 2;
  // printf("%d", partitionPageID);
  partition_t selectedPartitionPage = partitionPage[2];
  insertValue(&selectedPartitionPage, key, value);
}

void MR_Run(int argc, char *argv[], Mapper map, int num_mappers, 
      Reducer reduce, int num_reducers) {
  for(int i = 0; i < num_partition; ++i) {
    pthread_mutex_init(&(partitionPage[i].mutex), NULL);
    partitionPage[i].keyValue = NULL;
  }
  
  mapper_global = map;

  
  pthread_t map_thread[num_mappers];
  pthread_t reduce_thread[num_reducers];

  int fileTotalNumber = argc;
  int count = 1;
  
  // When the number of file is smaller than num_mappers.
  if (fileTotalNumber - 1 <= num_mappers)
    num_mappers = fileTotalNumber - 1;

  for(int i = 1; i <= num_mappers; ++i) {
    pthread_create(&map_thread[i - 1], NULL, start_mapper, (void*)argv[i]);
  }

  while(count < fileTotalNumber) {
    pthread_mutex_lock(&mutex);
    while (done == 0)
      pthread_cond_wait(&cond, &mutex);
    pthread_t new_map_thread;
    pthread_create(&new_map_thread, NULL, start_mapper,(void *)argv[count]);
    count++;
    pthread_mutex_unlock(&mutex);
  }

}
