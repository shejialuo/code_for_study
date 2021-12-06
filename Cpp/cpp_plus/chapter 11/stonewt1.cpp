#include <iostream>
using std::cout;

#include "stonewt1.h"

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

Stonewt::operator int() const {
  return int(pounds + 0.5);
}

Stonewt::operator double() const {
  return pounds;
}