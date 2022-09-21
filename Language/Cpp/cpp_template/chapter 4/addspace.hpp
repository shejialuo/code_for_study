#ifndef ADDSPACE_HPP
#define ADDSPACE_HPP

#include <iostream>

template<typename... Types>
void printWithoutSpace(const Types&... args) {
  (std::cout << ... << args) << '\n';
}

template <typename T>
class AddSpace {
private:
  const T& ref;
public:
  AddSpace(const T& r): ref(r) {}
  friend std::ostream& operator<<(std::ostream& os, AddSpace<T> s) {
    return os << s.ref << " ";
  }
};

template<typename... Types>
void printWithSpace(const Types&... args) {
  (std::cout << ... << AddSpace(args)) << "\n";
}

#endif