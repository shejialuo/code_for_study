#include "stack5decl.h"

template<typename T>
  template<typename T2>
Stack<T>& Stack<T>::operator=(Stack<T2> const& op2) {
  elems.clear();
  elems.insert(elems.begin(), op2.elems.cbegin(), op2.elems.cend());
  return *this;
}
