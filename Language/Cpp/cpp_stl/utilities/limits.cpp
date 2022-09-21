#include <iostream>
#include <limits>
#include <string>
using namespace std;

int main() {
  cout << boolalpha;

  cout << "max(float): " << numeric_limits<short>::max() << endl;
  cout << "max(int):   " << numeric_limits<int>::max() << endl;
  cout << "max(long):  " << numeric_limits<long>:: max() << endl;
  cout << endl;

  return 0;
}