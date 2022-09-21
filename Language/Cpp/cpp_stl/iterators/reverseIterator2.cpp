#include <iterator>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/*
  * This behavior is a consequence of the fact that
  * ranges are half open. To speicfy all elements of
  * a container, you must use the position after the
  * last element. However, for a reverse iterator,
  * this is the position before the element.
*/
int main() {
  vector<int> coll ={1,2,3,4,5,6,7,8,9};

  vector<int>::const_iterator pos;
  pos = find(coll.cbegin(), coll.cend(), 5);
  cout << "pos: " << *pos << endl;

  vector<int>::const_reverse_iterator rpos(pos);
  cout << "rpos: " << *rpos << endl;

  return 0;
}