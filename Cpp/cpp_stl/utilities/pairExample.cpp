#include <iostream>
#include <utility>
#include <tuple>
using namespace std;

class Foo {
public:
  Foo(tuple<int, float>) {
    cout << "Foo::Foo(tuple)" << endl;
  }
  template <typename... Args>
  Foo(Args... args) {
    cout << "Foo::Foo(args...)" << endl;
  }
};

int main() {
  tuple<int, float> t(1, 2.22);

  pair<int, Foo> p1(42, t);

  std::pair<int, float> pair1 = std::pair<int, float>(42, 7.77);
  std::pair<int, float> pair2 = std::make_pair(42, 7.77);

  return 0;
}
