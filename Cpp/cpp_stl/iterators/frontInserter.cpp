#include <list>
#include <algorithm>
#include <iterator>
#include <iostream>
using namespace std;

int main() {
  list<int> coll;

  front_insert_iterator<list<int>> iter(coll);

  *iter = 1;
  iter++;
  *iter = 2;
  iter++;
  *iter = 3;
  iter++;
  for(auto num : coll) {
    cout << num << " ";
  }
  cout << endl;

  front_inserter(coll) = 44;
  front_inserter(coll) = 55;
  for(auto num : coll) {
    cout << num << " ";
  }
  cout << endl;

  return 0;
}
