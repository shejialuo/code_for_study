#include <iostream>
#include <list>
#include <algorithm>
#include <iterator>
using namespace std;

int main() {
  list<int> coll = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  list<int>::iterator pos;
  int count = 0;
  pos = remove_if(coll.begin(), coll.end(),
                 [count](int) mutable {
                   return ++count == 3;
                 });
  coll.erase(pos, coll.end());

  copy(coll.cbegin(), coll.cend(),
       ostream_iterator<int>(cout, " "));
  cout << endl;

  return 0;
}
