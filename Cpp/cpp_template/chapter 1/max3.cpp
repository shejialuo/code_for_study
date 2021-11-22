#include <iostream>

/*
  * Ensure that all overloaded versions
  * of a function are declared before the
  * function is called
*/
template <typename T>
T max(T a, T b) {
  std::cout << "max<T>()\n";
  return b < a ? a : b;
}

template<typename T>
T max(T a, T b, T c) {
  return max(max(a,b), c); // uses the template version even for ints
                           // because the following declaration comes
                           // too late
}

int max(int a, int b) {
  std::cout << "max(int, int)\n";
  return b < a ? a : b;
}

int main() {
  ::max(47,11,33);
}