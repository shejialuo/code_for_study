#ifndef MAX_DEFAULT_HPP
#define MAX_DEFAULT_HPP

#include <type_traits>

template <typename T1, typename T2,
          typename RT = std::decay_t<decltype(true? T1{}: T2{})>>
RT max(T1 a, T2 b) {
  return b < a ? a : b;
}

#endif // MAX_DEFAULT_HPP