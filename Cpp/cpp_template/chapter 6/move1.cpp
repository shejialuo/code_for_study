#include <utility>
#include <iostream>

class X{

};

void g(X&) {
  std::cout << "g() for variable\n";
}

void