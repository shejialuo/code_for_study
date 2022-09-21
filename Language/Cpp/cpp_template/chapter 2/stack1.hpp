#ifndef STACK_1_HPP
#define STACK_1_HPP

#include <vector>
#include <cassert>

/*
  * The type of this class is Stack<T>,
  * with T being a template parameter.
  * You have to use Stack<T> whenever
  * you use the type of this class.
  * For example:
    ! Stack(Stack const&); == Stack (Stack<T>const &);
    ! Stack& operator=(Stack const&); == Stack<T>& operator=(Stack<T> const&);
*/

template <typename T>
class Stack {
private:
  std::vector<T> elems;
public:
  void push(T const& elem);
  void pop();
  const T& top() const;
  bool empty() const {
    return elems.empty();
  }
};

template <typename T>
void Stack<T>::push(const T& elem) {
  elems.push_back(elem);
}

template <typename T>
void Stack<T>::pop() {
  assert(!elems.empty());
  elems.pop_back();
}

template <typename T>
const T& Stack<T>::top() const {
  assert(!elems.empty());
  return elems.back();
}

#endif // STACK_1_HPP