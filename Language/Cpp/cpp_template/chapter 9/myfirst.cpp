#include <iostream>
#include <typeinfo>
#include "myfirst.hpp"

template<typename T>
void printTypeof(T const& x) {
  std::cout << typeid(x).name() << '\n';
}
