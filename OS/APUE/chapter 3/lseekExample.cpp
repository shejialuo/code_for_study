#include <unistd.h>
#include <fcntl.h>
#include <string>
using namespace std;

string buf1 = "abcdefghij";
string buf2 = "ABCDEFGHIJ";

int main() {
  int fd;

  fd = creat("file.hole", S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  write(fd, buf1.c_str(), 10);
  lseek(fd,16384, SEEK_SET);
  write(fd, buf2.c_str(), 10);

  return 0;
}