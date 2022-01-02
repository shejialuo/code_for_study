#include <sys/stat.h>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int i;
  struct stat buf;
  string s;
  for(int i = 1; i < argc; ++i) {
    cout << argv[i] << ": ";
    lstat(argv[i], &buf);
    if(S_ISREG(buf.st_mode)) {
      s = "regular";
    } else if(S_ISDIR(buf.st_mode)) {
      s = "directory";
    } else if(S_ISCHR(buf.st_mode)) {
       s = "character special";
    } else if(S_ISBLK(buf.st_mode)) {
       s = "block special";
    } else if(S_ISFIFO(buf.st_mode)) {
       s = "fifo";
    } else if(S_ISLNK(buf.st_mode)) {
       s = "symbolic link";
    } else if(S_ISSOCK(buf.st_mode)) {
       s = "socket";
    } else {
       s = "** unknown mode **";
    }
    cout << s << "\n";
  }

  return 0;
}