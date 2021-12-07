#ifndef PRINTCOLL_HPP
#define PRINTCOLL_HPP

/*
 * The keyword typename has to be used whenever a name
 * that depends on a template parameter is a type.
 * template <typename T>
 * class MyClass {
 *  public:
 *    void foo() {
 *      typename T::SubType* ptr;
 *    }
 * };
*/

#include <iostream>

template <typename T>
void printcoll(const T& coll) {
  typename T::const_iterator pos;
  typename T::const_iterator end(coll.end());
  for(pos=coll.begin(); pos != end; ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << '\n';
}

#endif // PRINTCOLL_HPP
