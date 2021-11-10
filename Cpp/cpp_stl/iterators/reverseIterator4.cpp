#include <iterator>
#include <iostream>
#include <list>
#include <algorithm>
using namespace std;

int main() {
  list<int> coll = {1,2,3,4,5,6,7,8,9};

  list<int>::const_iterator pos;
  pos = find(coll.cbegin(), coll.cend(), 5);

  list<int>::const_reverse_iterator rpos(pos);

  list<int>::const_iterator rrpos;
  rrpos = rpos.base();

  cout << "rrpos: " << *rrpos << endl;

  return 0;
}
