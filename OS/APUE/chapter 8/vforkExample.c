#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>

int globalVar = 6;

int main() {
  int var;
  pid_t pid;

  var = 88;
  printf("before fork\n");

  pid = vfork();
  if (pid < 0) {
    printf("fork error");
  } else if (pid == 0) {
    globalVar++;
    var++;
    _exit(0);
  }

  printf("pid = %ld, glob = %d, var = %d\n", (long)getpid(), globalVar,var);

  return 0;
}