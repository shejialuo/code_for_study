#include <array>
#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
using namespace std;

int main() {
  array<int, 10> coll = {11, 22, 33, 44};

  coll.back() = 9999;
  coll[coll.size() - 2] = 42;

  // tuple interface
  cout << get<0>(coll) << endl;

  return 0;
}
