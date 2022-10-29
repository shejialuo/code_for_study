#include <functional>
#include <string>
#include <iostream>

void printString(std::string const& s) {
  std::cout << s << '\n';
}

template<typename T>
void printT(T arg) {
  printString(arg);
}

int main() {
  std::string s = "hello";
  printT(s);
  printT(std::cref(s));
  return 0;
}
