#include <iostream>
#include <memory>

using namespace std;

int main() {

  /*
    * Error: two shared pointers manage allocated `int`
  */
  int *p = new int;
  shared_ptr<int> sp1(p);
  shared_ptr<int> sp2(p);

  /*
    * You should always directly initialize a smart pointer
    * the moment tou create the object with its associated resource.
  */
  shared_ptr<int> sp3(new int);
  shared_ptr<int> sp4(sp3);

  return 0;
}
