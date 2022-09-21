#include <iostream>
#include "stock00.h"

void Stock::acquire(const std::string& co, long n, double pr) {
  company = co;
  if (n < 0) {
    std::cout << "Number of shares can't be negative; "
              << company << " shares set to 0.\n";
  } else {
    shares = n;
  }

  shareVal = pr;
  setTot();
}

void Stock::buy(long num, double price) {
  if (num < 0) {
    std::cout << "Number of shares purchased can't be negative. "
              << "Transcation is aborted.\n";
  } else {
    shares += num;
    shareVal = price;
    setTot();
  }
}

void Stock::sell(long num, double price) {
  using std::cout;

  if(num < 0) {
    cout << "Number of shares sold can't be negative. "
         << "Transaction is aborted.\n";
  } else if(num > shares) {
    cout << "You can't sell more that you have! "
         << "Transaction is aborted.\n";
  } else {
    shares -= num;
    shareVal = price;
    setTot();
  }
}

void Stock::update(double price) {
  shareVal = price;
  setTot();
}

void Stock::show() {
  std::cout << "Company: " << company
            << "  Shares" << shares << '\n'
            << "  Share Price: $" << shareVal
            << "  Total Worth: $" << totalVal << '\n';
}