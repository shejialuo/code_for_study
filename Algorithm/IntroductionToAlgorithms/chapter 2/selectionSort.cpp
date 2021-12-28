#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

void selectionSort(vector<int>& data) {
  for(int i = 0; i < data.size(); ++i) {
    int minimal = i;
    for(int j = i + 1; j < data.size(); ++j) {
      if(data[j] < data[minimal]) {
        minimal = j;
      }
    }
    swap(data[i], data[minimal]);
  }
}

int main() {
  vector<int> data {1,3,2,1,5,1,2,3};
  selectionSort(data);
  copy(data.cbegin(), data.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}