#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>


int globalVar = 6;
char buf[] = "a write to stdout\n";

int main() {
  int var = 88;
  pid_t pid;

  if(write(STDOUT_FILENO, buf, sizeof(buf) - 1) != sizeof(buf) - 1)
    printf("write error\n");
  printf("before fork\n");

  if((pid = fork()) < 0) {
    printf("fork error");
  } else if (pid == 0) {
    globalVar++;
    var++;
  } else {
    sleep(2);
  }
  printf("pid = %ld, global = %d, var = %d\n",
         (long)getpid(), globalVar, var);
  exit(0);
}