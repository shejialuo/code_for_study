#include <unordered_set>
#include <numeric>
#include <iostream>
#include <iterator>
#include <algorithm>
using namespace std;

int main() {
  unordered_set<int> coll = {1,2,3,5,7,11,13,17,19,77};
  copy(coll.cbegin(), coll.cend(),
       ostream_iterator<int>(cout, " "));
  cout << endl;

  coll.insert({-7,17,33,-11,17,19,1,13});
  copy(coll.cbegin(), coll.cend(),
       ostream_iterator<int>(cout, " "));
  cout << endl;

  coll.erase(33);

  coll.insert(accumulate(coll.cbegin(), coll.cend(),0));
  copy(coll.cbegin(), coll.cend(),
       ostream_iterator<int>(cout, " "));
  cout << endl;

  return 0;
}