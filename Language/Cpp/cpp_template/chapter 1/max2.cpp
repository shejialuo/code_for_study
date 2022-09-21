#include <iostream>

/*
  * Overloading Function Templates
  * Rule1:
    * All other factors being equal, the overload
    * resolution process prefers the nontemplate.
  * Rule2:
    * If the template can generate a function with
    * a better match, the template is selected.
  * Rule3:
    * Automatic type conversion is not considered
    * for deduced template arguments but is considered
    * for ordinary function paramteters. So the
    * max('a', 42.7) use nontemplate one.
  ! Pay attention: you should ensure that only one
  ! of them match for each call.
*/
int max(int a, int b) {
  std::cout << "call nontemplate function\n";
  return b < a ? a : b;
}

template<typename T>
T max(T a, T b) {
  std::cout << "call template function\n";
  return b < a ? a : b;
}

int main() {
  ::max(7,42);          // calls the nontemplate for two ints
  ::max(7.0,42.0);      // calls max<double>(by argument deduction)
  ::max('a', 'b');      // calls max<char>(by argument deduction)
  ::max<>(7,42);        // calls max<int>(by argument deduction)
  ::max<double>(7,42);  // calls max<double>(no argument deduction)
  ::max('a', 42.7);     // calls the nontemplate for two ints
}