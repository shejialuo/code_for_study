#include <set>
#include <list>
#include <algorithm>
#include <iterator>
#include <iostream>
using namespace std;

int main() {
  set<int> coll;

  insert_iterator<set<int>> iter(coll, coll.begin());

  *iter = 1;
  iter++;
  *iter = 2;
  iter++;
  *iter = 3;

  for(auto num : coll) {
    cout << num << " ";
  }
  cout << endl;

  inserter(coll, coll.end()) = 44;
  inserter(coll, coll.end()) = 55;

  list<int> coll2;
  copy(coll.begin(), coll.end(),
       inserter(coll2, coll2.begin()));

  for(auto num : coll2) {
    cout << num << " ";
  }
  cout << endl;

  return 0;
}
