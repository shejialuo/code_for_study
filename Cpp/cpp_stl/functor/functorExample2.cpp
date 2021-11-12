#include <iostream>
#include <list>
#include <algorithm>
#include <iterator>
using namespace std;

class IntSequence {
private:
  int value;
public:
  IntSequence(int initialValue): value(initialValue) {}

  int operator() () {
    return ++value;
  }
};

int main() {
  list<int> coll;

  generate_n(back_inserter(coll), 9, IntSequence(1));
  copy(coll.cbegin(), coll.cend(),
       ostream_iterator<int>(cout, " "));
  cout << endl;

  generate(next(coll.begin()), prev(coll.end()), IntSequence(42));

  copy(coll.cbegin(), coll.cend(),
       ostream_iterator<int>(cout, " "));
  cout << endl;

  return 0;
}