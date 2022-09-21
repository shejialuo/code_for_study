#ifndef STACK_3_HPP
#define STACK_3_HPP

#include <vector>
#include <cassert>

/*
  * You can define default values
*/

template <typename T, typename Cont = std::vector<T>>
class Stack {
private:
  Cont elems;
public:
  void push(T const& elem);
  void pop();
  const T& top() const;
  bool empty() const {
    return elems.empty();
  }
};

template <typename T, typename Cont>
void Stack<T, Cont>::push(const T& elem) {
  elems.push_back(elem);
}

template <typename T, typename Cont>
void Stack<T, Cont>::pop() {
  assert(!elems.empty());
  elems.pop_back();
}

template <typename T, typename Cont>
const T& Stack<T, Cont>::top() const {
  assert(!elems.empty());
  return elems.back();
}

#endif // STACK_3_HPP