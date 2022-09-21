#ifndef STACK_2_HPP
#define STACK_2_HPP

#include "stack1.hpp"
#include <deque>
#include <string>
#include <cassert>

/*
  * You can specialize a class template for certain template arguments.
*/

template<>
class Stack<std::string> {
private:
  std::deque<std::string> elems;
public:
  void push(const std::string&);
  void pop();
  const std::string& top() const;
  bool empty() const {
    return elems.empty();
  }
};

void Stack<std::string>::push(const std::string& elem) {
  elems.push_back(elem);
}

void Stack<std::string>::pop() {
  assert(!elems.empty());
  elems.pop_back();
}

const std::string& Stack<std::string>::top() const {
  assert(!elems.empty());
  return elems.back();
}

#endif // STACK_2_HPP