#ifndef MYTIME2_H_
#define MYTIME2_H_

#include <iostream>

class Time {
private:
  int hours;
  int minutes;
public:
  Time();
  Time(int h, int m = 0);
  void addMin(int m);
  void addHr(int h);
  void reset(int h = 0, int m = 0);
  Time operator*(double n) const;
  void show() const;
  friend Time operator*(double m, const Time& t) {
    return t * m;
  }
  friend std::ostream& operator<<(std::ostream& os, const Time& t);
};

#endif // MYTIME2_H_