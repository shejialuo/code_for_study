#include <functional>
#include <iostream>
using namespace std;
using namespace std::placeholders;

int main() {
  auto plus10 = bind(plus<int>(), _1, 10);
  cout << "+10: " << plus10(7) << endl;

  auto plus10times2 = bind(multiplies<int>(),
                           bind(plus<int>(), _1, 10),2);
  cout << "+10 * 2: " << plus10times2(7) << endl;

  return 0;
}