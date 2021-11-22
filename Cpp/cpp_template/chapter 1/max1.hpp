#ifndef MAX1_HPP
#define MAX1_HPP

/*
  * T is a template parameter.
  * The keyword typename introduces a type parameter
  * T should be copyable and support operator <
*/

template <typename T>
T max(T a, T b) {
  return b < a ? a : b;
}

#endif // MAX1_HPP