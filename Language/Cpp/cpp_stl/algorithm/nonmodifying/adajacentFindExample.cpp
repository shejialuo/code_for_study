#include "../algostuff.hpp"

using namespace std;

bool doubled(int elem1, int elem2) { return elem1 * 2 == elem2; }

int main() {
  vector<int> coll;

  coll.push_back(1);
  coll.push_back(3);
  coll.push_back(2);
  coll.push_back(4);
  coll.push_back(5);
  coll.push_back(5);
  coll.push_back(0);

  copy(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  vector<int>::iterator pos;
  pos = adjacent_find(coll.begin(), coll.end());

  if (pos != coll.end()) {
    cout << "first adjacent pair of equal elements at: "
         << distance(coll.begin(), pos) + 1 << "\n";
  }

  pos = adjacent_find(coll.begin(), coll.end(), doubled);
  if (pos != coll.end()) {
    cout << "first adjacent pair of elements with the second value twice the "
            "first value at: "
         << distance(coll.begin(), pos) + 1 << "\n";
  }

  return 0;
}
