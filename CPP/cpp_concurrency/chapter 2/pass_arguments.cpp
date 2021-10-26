#include <iostream>
#include <thread>
#include <string>
using namespace std;

void func(int i, const string& s) {
  cout << i << " " << s << "\n";
}

/*
  It's a bad iead to use a pointer to an automatic variable
*/
void opps(int param) {
  char buffer[1024];
  sprintf(buffer, "%i", param);
  /*
    * buffer is const char*, and it needs to be changed
    * to string, however, the buffer may be destroyed
    * before this completes
  */
  thread t(func, 3, buffer);
  t.detach();
}

void fixOpps(int param) {
  char buffer[1024];
  sprintf(buffer,"%i", param);
  // explict convert const char* to string.
  thread t(func, 3, string(buffer));
  t.detach();
}

int main() {
  opps(5);
  fixOpps(5);
  return 0;
}