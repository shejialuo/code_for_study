#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int> v;
  unsigned sz = v.size();

  /*
   * The official return type of v.size() is
   * vector<int>::size_type.
  */

  auto size = v.size();    // better
}
