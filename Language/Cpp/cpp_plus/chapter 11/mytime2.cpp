#include <iostream>
#include "mytime2.h"

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

Time Time::operator*(double n) const {
  Time result;
  long totalminutes = hours * n * 60 + minutes * n;
  result.hours = totalminutes / 60;
  result.minutes = totalminutes % 60;
  return result;
}

void Time::show() const {
  std::cout << hours << "hours, " << minutes << " minutes";
}

std::ostream& operator<<(std::ostream& os, const Time& t) {
  os << t.hours << " hours, " << t.minutes << " minutes";
  return os;
}