#include <iostream>
#include "mytime1.h"

/*
  * Overload restrictions
  * 1. The overload operator must have at least one
  *    operand that is a user-defined types.
  * 2. You can't use an operator in a manner that violates
  *    the syntax rules for the original operator.
  * 3. You can't create new operator symbols
  * 4. You can't overload the following operators:
  *    + sizeof
  *    + .
  *    + .*
  *    + ::
  *    + ?:
  *    + typeid
  *    + const_cast
  *    + dynamic_cast
  *    + reinterpret_cast
  *    + static_cast
  * 5. Most of operators can be overloaded by using
  *    either member of nonmember functions. However,
  *    you can use only member functions to overload
  *    the following opeators:
  *    =, (), [] and ->
*/
Time::Time() {
  hours = minutes = 0;
}

Time::Time(int h, int m) {
  hours = h;
  minutes = m;
}

void Time::addMin(int m) {
  minutes += m;
  hours += minutes / 60;
  minutes %= 60;
}

void Time::addHr(int h) {
  hours += h;
}

void Time::reset(int h, int m) {
  hours = h;
  minutes = m;
}

Time Time::operator+(const Time& t) const {
  Time sum;
  sum.minutes = minutes + t.minutes;
  sum.hours = hours + t.hours + sum.minutes / 60;
  sum.minutes %= 60;
  return sum;
}

void Time::show() const {
  std::cout << hours << "hours, " << minutes << " minutes";
}
