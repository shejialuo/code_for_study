#include "../algostuff.hpp"
using namespace std;

int main() {
  vector<int> coll;

  INSERT_ELEMENTS(coll,1,9);
  PRINT_ELEMENTS(coll, "coll: ");

  rotate(coll.begin(), coll.begin() + 1 , coll.end());
  PRINT_ELEMENTS(coll, "one left: ");

  rotate(coll.begin(), coll.end() - 2, coll.end());
  PRINT_ELEMENTS(coll, "two right: ");

  vector<int>::const_iterator pos = next(coll.cbegin());
  rotate_copy(coll.cbegin(), pos, coll.cend(),
            ostream_iterator<int>(cout, " "));
  cout << endl;

  return 0;
}
