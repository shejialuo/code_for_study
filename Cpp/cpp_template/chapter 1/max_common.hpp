#ifndef MAX_COMMON_HPP
#define MAX_COMMON_HPP

#include <type_traits>

//* Declare the return type to be the common type
//* Since c++14
template <typename T1, typename T2>
std::common_type_t<T1, T2> max(T1 a, T2 b) {
  return b < a ? a : b;
}

#endif // MAX_COMMON_HPP
