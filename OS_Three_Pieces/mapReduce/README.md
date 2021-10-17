# Map Reduce

This is a MapReduce for studying. Actually it is the
project work of Operating Systems Three Easy Pieces
[MapReduce](https://github.com/remzi-arpacidusseau/ostep-projects)
You can find detail here.

## Limitations

However, I only do the map operation. However, I think the most
important thing is the concurrent data structure.

```c
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
```
