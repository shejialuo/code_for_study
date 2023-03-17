#include "../algostuff.hpp"

using namespace std;

int main() {
  vector<int> coll{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  copy(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  remove_copy(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "), 3);
  cout  << "\n";

  remove_copy_if(coll.cbegin(), coll.cend(), ostream_iterator<int>(cout, " "),
    [](int elem) { return elem < 4; });
  cout << "\n";

  return 0;
}
