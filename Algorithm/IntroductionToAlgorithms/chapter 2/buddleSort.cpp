#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

void buddleSort(vector<int>& data) {
  for(int i = 0; i < data.size(); ++i) {
    for(int j = 0; j < data.size() - i - 1; ++ j) {
      if(data[j] > data[j + 1]) {
        swap(data[j], data[j + 1]);
      }
    }
  }
}

int main() {
  vector<int> data {1,3,2,1,5,1,2,3};
  buddleSort(data);
  copy(data.cbegin(), data.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}