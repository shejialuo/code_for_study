#include "stack1.hpp"
#include <iostream>
#include <string>

/*
  * To use an object of a template class,
  * until C++17 you must always specify the
  * the template arguments explicitly.
*/

/*
  * Code is instantiated only for template
  * member functions that are called.
*/
int main() {
  Stack<int> intStack;
  Stack<std::string> stringStack;

  intStack.push(7);
  std::cout << intStack.top() << '\n';

  stringStack.push("hello");
  std::cout << stringStack.top() << '\n';
  stringStack.pop();

  return 0;
}