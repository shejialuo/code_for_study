#ifndef STONEWT_H_
#define STONEWT_H_

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
};

#endif // STONEWT_H_