#include <iostream>
#include "varprint1.hpp"

template<typename... T>
void printDoubled(T... args) {
  print((args + args)...);
}

template<typename... T>
void addOne(T... args) {
  print((args + 1)...);
}

int main() {
  printDoubled(1,2,3,4,5,6);
  addOne(1,2,3,4,5,6);
  return 0;
}
