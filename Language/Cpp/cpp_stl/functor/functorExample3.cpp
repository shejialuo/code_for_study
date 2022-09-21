#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class MeanValue {
private:
  long num;
  long sum;
public:
  MeanValue(): num(0), sum(0) {}

  void operator()(int elem) {
    ++num;
    sum += elem;
  }

  double value() {
    return static_cast<double>(sum) / static_cast<double>(num);
  }
};

int main() {
  vector<int> coll = {1, 2, 3, 4, 5, 6, 7, 8};
  /*
    * Note that lambdas provide a more convenient way to specify
    * this behavior. However, that does not mean that lambdas are
    * always better than function objects. Function objects
    * are more convenient when their type is required, such as for
    * a declaration of a hash function, sorting, or equivalence
    * criterion of associative or unordered containers.
  */
  MeanValue mv = for_each(coll.begin(), coll.end(), MeanValue());
  cout << "mean value: " << mv.value() << endl;
  return 0;
}
