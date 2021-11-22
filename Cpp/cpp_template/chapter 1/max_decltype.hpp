#ifndef MAX_DECLTYPE_HPP
#define MAX_DECLTYPE_HPP

#include <type_traits>

//* Use trailing return type
template <typename T1, typename T2>
auto maxNotDecay(T1 a, T2 b) -> decltype(b<a?a:b) {
  return b < a ? a : b;
}

//! However, it might happen the return type is a reference.

template <typename T1, typename T2>
auto maxDecay(T1 a, T2 b) ->
  typename std::decay<decltype(b<a?a:b)>::type {
  return b < a ? a : b
}

#endif // MAX_DECLTYPE_HPP