
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

#define DEF_PAGER "/bin/more"

int main(int argc, char *argv[]) {
  int n;
  int fd[2];
  pid_t pid;
  char *pager, *argv0;
  char line[4096];
  FILE *fp;

  if (argc != 2) {
    fprintf(stdout, "usage: a.out <pathname>\n");
    return 0;
  }

  fp = fopen(argv[1], "r");
  pipe(fd);

  pid = fork();
  if (pid > 0)  {
    close(fd[0]);

    while (fgets(line, 4096, fp) != NULL) {
      n = strlen(line);
      if (write(fd[1], line , n) != n) {
        fprintf(stderr, "error\n");
      }
    }
    close(fd[1]);
    waitpid(pid, NULL, 0);
  } else {
    close(fd[1]);
    if (fd[0] != STDIN_FILENO) {
      dup2(fd[0], STDIN_FILENO);
      close(fd[0]);
    }

    pager = DEF_PAGER;
    if ((argv0 = strrchr(pager, '/')) != NULL) {
      argv0++;
    } else {
      argv0 = pager;
    }

    execl(pager, argv0, (char*)0);

  }
  return 0;
}
