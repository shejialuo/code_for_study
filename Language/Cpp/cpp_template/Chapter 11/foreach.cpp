#include <iostream>
#include <vector>
#include "foreach.hpp"

void func(int i) {
  std::cout << "func() called for: " << i << '\n';
}

class FuncObj {
public:
  void operator() (int i) const {
    std::cout << "FuncObj::op() called for: " << i << '\n';
  }
};

int main() {
  std::vector<int> primes {2, 3, 5, 7, 11, 13, 17, 19};
  // function as callable (decays to pointer)
  foreach(primes.begin(), primes.end(), func);
  // function pointer as callable
  foreach(primes.begin(), primes.end(), &func);
  // function object as callable
  foreach(primes.begin(), primes.end(), FuncObj());
  // lambda as callable
  foreach(primes.begin(), primes.end(), [](int i) {
    std::cout << "lambda called for: " << i << '\n';
  });
}
