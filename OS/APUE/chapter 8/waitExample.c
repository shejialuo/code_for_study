#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

int main() {
  pid_t pid;

  printf("before fork\n");

  pid = fork();
  if (pid < 0) {
    printf("fork error");
  } else if (pid == 0) {
    pid = fork();
    if (pid < 0) {
      printf("fork error");
    } else if (pid > 0) {
      exit(0);
    }
    /*
     * We're the second child; our parent become init
     */
    sleep(2);
    printf("second child, parent pid = %ld\n", (long)getppid());
    exit(0);

  }

  if(waitpid(pid, NULL, 0) != pid) {
    printf("waitpid error");
  }

  exit(0);
}