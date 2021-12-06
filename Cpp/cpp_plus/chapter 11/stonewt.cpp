#include <iostream>
using std::cout;

#include "stonewt.h"

Stonewt::Stonewt(double lbs) {
  stone = int(lbs) / lbsPerStn;
  pdsLeft = int(lbs) % lbsPerStn + lbs - int(lbs);
  pounds = lbs;
}

Stonewt::Stonewt(int stn, double lbs) {
  stone = stn;
  pdsLeft = lbs;
  pounds = stn * lbsPerStn + lbs;
}

Stonewt::Stonewt() {
  stone = pounds = pdsLeft = 0;
}

Stonewt::~Stonewt() {}

void Stonewt::showStn() const {
  cout << stone << " stone, " << pdsLeft << " pounds\n";
}

void Stonewt::showLbs() const {
  cout << pounds << " pounds\n";
}

/*
  * Only a constructor that can be used with just
  * one argument works as a conversion function.
*/
Stonewt myCat = 19.6; /* Bad way*/

/*
  * It is used for the following implicit conversions:
  * 1. When you initialize a Stonewt object to a type double value
  * 2. When you assign a type double value to a Stonewt value
  * 3. When you pass a type double value to a function that
  *    excepts a Stonewt argument
  * 4. When a function that's declared to return a Stonewt value
  *    tries to return a double value
*/

