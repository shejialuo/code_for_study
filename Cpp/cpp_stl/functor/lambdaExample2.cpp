#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
  vector<int> coll = {1, 2, 3, 4, 5, 6, 7, 8};

  long sum = 0;
  for_each(coll.begin(), coll.end(), [&sum](int elem) {
    sum += elem;
  });
  /*
    * Here, instead of the need to define a class for the
    * function object passed, you simply pass the required
    * functionality. However, the state of the calculation
    * is held outside the lambda in `sum`, so you ultimately
    * have to use `sum` to compute the mean value.
    *
    * With a function object, this state is entirely
    * encapsulated, and we can provide additional member
    * functions to deal with the state.
  */
  double mv = static_cast<double>(sum)/static_cast<double>(coll.size());
  cout << "meav value: " << mv << endl;

  return 0;
}
