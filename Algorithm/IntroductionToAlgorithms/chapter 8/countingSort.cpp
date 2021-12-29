#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
using namespace std;

void countingSort(vector<int>& data,vector<int>& sortedData, int k) {
  int auxArray[k + 1];
  for(int i = 0; i <= k; ++i) {
    auxArray[i] = 0;
  }
  for(int i = 0; i < data.size(); ++i) {
    auxArray[data[i]]++;
  }
  for(int i = 1; i <= k; ++i) {
    auxArray[i] += auxArray[i - 1];
  }
  for(int i = data.size() - 1; i >= 0; --i) {
    sortedData[auxArray[data[i]] - 1] = data[i];
    auxArray[data[i]]--;
  }
}

int main() {
  vector<int> data {2,5,3,0,2,3,0,3};
  vector<int> sortedData(data.size());
  countingSort(data, sortedData,
               *max_element(data.cbegin(), data.cend()));
  copy(sortedData.cbegin(), sortedData.cend(),
       ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}