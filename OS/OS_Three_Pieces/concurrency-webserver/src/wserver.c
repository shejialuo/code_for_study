#include <stdio.h>
#include <pthread.h>
#include "request.h"
#include "io_helper.h"

char default_root[] = ".";

//
// ./wserver [-d basedir] [-p port] [-t threads] [-b buffers] [-s schedalg]
//

int bufferP = -1;
pthread_mutex_t mutex;
pthread_cond_t full;
pthread_cond_t empty;

typedef struct thread_pool thread_pool_t;

struct thread_pool {
    int *buffers;
    pthread_t *pool;
};

void *request(void *arg) {
    thread_pool_t *threadPool = (thread_pool_t *)arg;
    int conn_fd;
    pthread_mutex_lock(&mutex);
    while(bufferP == -1)
        pthread_cond_wait(&full,&mutex);
    conn_fd = threadPool->buffers[bufferP];
    bufferP--;
    pthread_cond_signal(&empty);
    pthread_mutex_unlock(&mutex);

    request_handle(conn_fd);
	close_or_die(conn_fd);

    return (void*)0;
}

int main(int argc, char *argv[]) {
    int c;
    char *root_dir = default_root;
    int port = 10000;
    int threads = 1;
    int buffers = 1;
    char *schedalg = "FIFO";

    // fprintf(stdout, "dasdas");

    while ((c = getopt(argc, argv, "d:p:t:b:s:")) != -1)
	switch (c) {
	case 'd':
	    root_dir = optarg;
	    break;
	case 'p':
	    port = atoi(optarg);
	    break;
    case 't':
        threads = atoi(optarg);
        break;
    case 'b':
        buffers = atoi(optarg);
        break;
    case 's':
        schedalg = (char *)malloc(strlen(optarg) + 1);
        strcpy(schedalg, optarg);
        break;
	default:
	    fprintf(stderr, "usage: ./wserver [-d basedir] [-p port] [-t threads] [-b buffers] [-s schedalg]\n");
	    exit(1);
	}

    // run out of this directory
    chdir_or_die(root_dir);
    // now, get to work
    int listen_fd = open_listen_fd_or_die(port);

    thread_pool_t *threadPool = (thread_pool_t *)malloc(sizeof(thread_pool_t));
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&full, NULL);
    pthread_cond_init(&empty, NULL);
    threadPool->buffers = (int *)malloc(buffers*sizeof(int));
    threadPool->pool = (pthread_t *)malloc(threads*sizeof(pthread_t));
    for(int i = 0; i < threads; ++i) {
        pthread_create(&threadPool->pool[i], NULL, request, (void*)threadPool);
    }

    while (1) {
	    struct sockaddr_in client_addr;
	    int client_len = sizeof(client_addr);
	    int conn_fd = accept_or_die(listen_fd, (sockaddr_t *) &client_addr, (socklen_t *) &client_len);
        pthread_mutex_lock(&mutex);
        while (bufferP == buffers - 1)
            pthread_cond_wait(&empty, &mutex);

        bufferP++;
        threadPool->buffers[bufferP] = conn_fd;

        pthread_cond_signal(&full);
        pthread_mutex_unlock(&mutex);
    }
    return 0;
}
