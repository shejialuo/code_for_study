#include <iostream>

template<typename... Types>
int foldl(int initialValue, Types... args) {
  return (initialValue + ... + args);
}

template<typename... Types>
int foldr(int initialValue, Types... args) {
  return (args + ... + initialValue);
}

int main() {
  std::cout << foldl(1,2,3,4,5) << '\n';
  std::cout << foldr(1,2,3,4,5) << "\n";
  return 0;
}
