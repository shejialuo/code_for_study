#include "max1.hpp"
#include <iostream>
#include <string>

/*
  * Templates aren't compiled into single entites that can
  * handle any type. Instead, different entities are
  * generated from the template for every type for
  * which the template is used.
*/

/*
  * The process of replacing template parameters by concrete
  * types is called instantiation.
*/

int main() {
  int i = 42;
  std::cout << "max(7,i):   " << ::max(7,i) << "\n";

  double f1 = 3.4;
  double f2 = -6.7;
  std::cout << "max(f1,f2): " << ::max(f1,f2) << "\n";

  std::string s1 = "mathematics";
  std::string s2 = "math";
  std::cout << "max(s1,s2): " << ::max(s1,s2) << "\n";

  return 0;
}