#include <stdio.h>
#include <unistd.h>


int main(int argc, char* argv[]) {
  int n;
  int fd[2];
  pid_t pid;

  char line[4096];

  if (pipe(fd) < 0) {
    fprintf(stderr, "pipe error");
  }

  if ((pid = fork()) < 0) {
    fprintf(stderr, "fork error");
  } else if (pid > 0) {
    close(fd[0]);
    write(fd[1], "hello world\n", 12);
  } else {
    close(fd[1]);
    n = read(fd[0], line, 4096);
    write(STDOUT_FILENO, line, n);
  }
  return 0;
}