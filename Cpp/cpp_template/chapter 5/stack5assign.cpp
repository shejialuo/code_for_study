#include "stack5decl.h"

template<typename T>
  template<typename T2>
Stack<T>& Stack<T>::operator=(Stack<T2> const& op2) {
  Stack<T2> temp(op2);
  elems.clear();
  while(!temp.empty()) {
    elems.push_front(temp.top());
    temp.pop();
  }
  return *this;
}