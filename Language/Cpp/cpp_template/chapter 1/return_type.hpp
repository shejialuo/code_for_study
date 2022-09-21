#ifndef RETURN_TYPE_HPP
#define RETURN_TYPE_HPP

//* Introduce a third template parameter for the return type.
template <typename RT, typename T1, typename T2>

RT max(T1 a, T2 b) {
  return b < a ? a : b;
}

#endif // RETURN_TYPE_HPP