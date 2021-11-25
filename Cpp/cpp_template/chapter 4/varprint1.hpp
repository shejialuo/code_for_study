#ifndef VARPRINT1_HPP
#define VARPRINT1_HPP

/*
  * Template parameters can be defined to accept an unbounded
  * number of template arguments. Templates with this ability
  * are called variadic templates
*/

#include <iostream>

//! End recursion
void print() {}

template <typename T, typename... Types>
void print(T firstArg, Types... args) {
  std::cout << firstArg << "\n";
  print(args...);
}

/*
  * It is just recursive. So we need to define
  * void print();
*/

#endif // VARPRINT1_HPP