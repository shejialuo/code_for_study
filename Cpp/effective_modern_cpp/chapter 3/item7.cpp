#include <initializer_list>
#include <iostream>

class Widget {
private:
  int x { 0 };
  int y = 0;
  // int z(0) would error!;
public:
  Widget() {}
  Widget(int i, bool b) {}
  Widget(int i , double d) {}
  Widget(std::initializer_list<long double> il) {}
};


int main() {
  int x(0);         // initializer is parentheses
  int y = 0;        // initializer follows "="
  int z1 { 0 };     // initializer is in braces
  int z2 = { 0 };   // initializer uses "=" and braces

  Widget w1;       // call default constructor
  Widget w2 = w1;  // not an assignment; calls copy constructor
  w1 = w2;         // an assignment; calls copy operator=

  Widget w3();     // most vexing parse! declares a function
                   // named w3 that retuns a Widget!

  Widget w4{};

  Widget w5(10, true);   // calls first constructor

  Widget w6{10, true};   // calls std::initializer_list constructor

  return 0;
}
