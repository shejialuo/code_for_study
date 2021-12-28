#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

void insertionSort(vector<int>& data) {
  for(int i = 1; i < data.size(); ++i) {
    int pivot = data[i];
    int j = i - 1;
    while(j >= 0 && data[j] > pivot) {
      data[j + 1] = data[j];
      j--;
    }
    data[j + 1] = pivot;
  }
}

int main() {
  vector<int> data {1,3,2,1,5,1,2,3};
  insertionSort(data);
  copy(data.cbegin(), data.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}