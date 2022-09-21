#include "return_type.hpp"
#include <iostream>

int main() {
  std::cout << "max(4,7.2): " << ::max<double>(4, 7.2);
  return 0;
}