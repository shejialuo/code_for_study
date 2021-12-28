#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
using namespace std;

void merge(vector<int>& data, int p, int m, int r) {
  int i = 0, j = 0;
  int lengthFormer = m - p + 1;
  int lengthLatter = r - m ;
  int tempArrayFormer[lengthFormer + 1];
  int tempArrayLatter[lengthLatter + 1];
  while(i < lengthFormer) {
    tempArrayFormer[i] = data[p + i];
    i++;
  }
  tempArrayFormer[i] = numeric_limits<int>::max();
  while(j < lengthLatter) {
    tempArrayLatter[j] = data[m + j + 1];
    j++;
  }
  tempArrayLatter[j] = numeric_limits<int>::max();
  
  i = 0;
  j = 0;
  for(int k = p; k <= r; ++k) {
    if(tempArrayFormer[i] <= tempArrayLatter[j]) {
      data[k] = tempArrayFormer[i++];
    } else {
      data[k] = tempArrayLatter[j++];
    }
  }

}

void mergeSort(vector<int>& data, int p, int r) {
  if(p < r) {
    int mid = (p + r) / 2;
    mergeSort(data, p, mid);
    mergeSort(data, mid + 1, r);
    merge(data, p, mid ,r);
  }
}

int main() {
  vector<int> data {1,3,2,1,5,1,2,3};
  mergeSort(data, 0, data.size() - 1);
  copy(data.cbegin(), data.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}