#include <iostream>

struct Test {
  Test() { std::cout << "Test::Test\n"; }
  Test(const Test &) { std::cout << "Test(const Test&)\n"; }
  Test(Test &&) { std::cout << "Test(Test&&)\n"; }
  ~Test() { std::cout << "~Test\n"; }
};

Test Create() {
  return Test();
}

int main() {
  auto n = Create();
}
