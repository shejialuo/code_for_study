#ifndef STOCK10_H_
#define STOCK10_H_

#include <string>

class Stock {
private:
  std::string company;
  long shares;
  double shareVal;
  double totalVal;
  void setTot() {totalVal = shares * shareVal;}
public:
  Stock();
  Stock(const std::string& co, long n = 0, double pr = 0.0);
  ~Stock();
  void buy(long num, double price);
  void sell(long num, double price);
  void update(double price);
  void show();
};

#endif // STOCK10_H_