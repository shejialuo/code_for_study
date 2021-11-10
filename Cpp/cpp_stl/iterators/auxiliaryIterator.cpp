#include <iterator>
#include <iostream>
#include <list>
#include <algorithm>
using namespace std;

int main() {
  list<int> coll;

  for(int i = 1; i <= 9; ++i) {
    coll.push_back(i);
  }

  list<int>::iterator pos = coll.begin();
  cout << *pos << endl;

  advance(pos, 3);
  cout << *pos << endl;

  advance(pos, -1);
  cout << *pos << endl;

  next(pos);
  cout << *pos << endl;

  next(pos, 2);
  cout << *pos << endl;

  next(pos, -2);
  cout << *pos << endl;

  prev(pos);
  cout << *pos << endl;

  cout << distance(coll.begin(), pos) << endl;

  iter_swap(coll.begin(), next(coll.begin()));

  cout << *coll.begin() << " " << *next(coll.begin()) << endl;

  return 0;
}
