#include "../algostuff.hpp"

using namespace std;

int main() {
  vector<int> coll{2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7};

  copy(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  auto pos = remove(coll.begin(), coll.end(), 5);
  coll.erase(pos, coll.end());

  copy(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  pos = remove_if(coll.begin(), coll.end(), [](const int elem) { return elem < 4;});
  coll.erase(pos, coll.end());

  copy(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  return 0;
}