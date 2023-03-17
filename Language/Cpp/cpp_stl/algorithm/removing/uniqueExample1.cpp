#include "../algostuff.hpp"

using namespace std;

int main() {
  vector<int> coll1{1, 4, 4, 6, 1, 2, 2, 3, 1, 6, 6, 6, 5, 7, 5, 4, 4};

  // unique removes all elements that are equal to the previous elements.
  // Thus only when the elements in the sequence are sorted or at least
  // when all elements of the same value are adjacent, does it remove all duplicates

  auto pos = unique(coll1.begin(), coll1.end());
  copy(coll1.cbegin(), coll1.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  vector<int> coll2{1, 4, 4, 6, 1, 2, 2, 3, 1, 6, 6, 6, 5, 7, 5, 4, 4};
  pos = unique(coll1.begin(), coll1.end(), greater<int>());
  copy(coll1.cbegin(), coll1.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  return 0;
}
