#ifndef MAX_AUTO_HPP
#define MAX_AUTO_HPP

//* Let the compiler find out since c++14
template <typename T1, typename T2>
auto max(T1 a, T2 b) {
  return b < a ? a : b;
}

#endif // MAX_AUTO_HPP