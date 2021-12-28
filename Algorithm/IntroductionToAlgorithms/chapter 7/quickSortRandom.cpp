#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <random>
using namespace std;

int partition(vector<int>& data, int p, int r) {
  int random = (rand() % (r - p + 1)) + p;
  swap(data[random], data[r]);
  int pivot = data[r];
  int i = p, j = p - 1;
  for( ;i < r; ++i) {
    if(data[i] < pivot) {
      swap(data[i], data[++j]);
    }
  }
  swap(data[j + 1], data[r]);
  return j + 1;
}

void quickSortRandom(vector<int>& data, int p ,int r) {
  if(p < r) {
    int q = partition(data, p , r);
    quickSortRandom(data, p , q - 1);
    quickSortRandom(data, q + 1, r);
  }
}

int main() {
  vector<int> data {1,3,2,1,5,1,2,3};
  quickSortRandom(data, 0 ,data.size() - 1);
  copy(data.cbegin(), data.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}
