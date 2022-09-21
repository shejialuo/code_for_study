#ifndef STACK_NOTYPE_HPP
#define STACK_NOTYPE_HPP

#include <array>
#include <cassert>

template<typename T, std::size_t Maxsize>
class Stack {
private:
  std::array<T, Maxsize> elems;
  std::size_t numElems;
public:
  Stack();
  void push(const T& elem);
  void pop();
  const T& top() const;
  bool empty() const {
    return numElems == 0;
  }
  std::size_t size() const {
    return numElems;
  }
};

template <typename T, std::size_t Maxsize>
Stack<T, Maxsize>::Stack(): numElems(0) {}

template <typename T, std::size_t Maxsize>
void Stack<T, Maxsize>::push(const T& elem) {
  assert(numElems < Maxsize);
  elems[numElems] = elem;
  ++numElems;
}

template <typename T, std::size_t MaxSize>
void Stack<T, MaxSize>::pop() {
  assert(!elems.empty());
  --numElems;
}

template <typename T, std::size_t MaxSize>
const T& Stack<T, MaxSize>::top() const {
  assert(!elems.empty());
  return elems[numElems - 1];
}

#endif // STACK_NOTYPE_HPP