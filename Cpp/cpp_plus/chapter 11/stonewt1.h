#ifndef STONEWT1_H_
#define STONEWT1_H_

/*
  * You can use a special form of a C++ operator function
  * called conversion function to convert a object to a
  * type.
  * Syntax: operator typeName();
  * Note:
  * 1. The conversion function must be a class method
  * 2. The conversion function must not specify a return type.
  * 3. The conversion function must have no arguments.
*/
class Stonewt {
private:
  enum {lbsPerStn = 14};
  int stone;
  double pdsLeft;
  double pounds;
public:
  Stonewt(double lbs);
  Stonewt(int stn, double lbs);
  Stonewt();
  ~Stonewt();
  void showLbs() const;
  void showStn() const;
  operator int() const;
  operator double() const;
};

#endif // STONEWT1_H_