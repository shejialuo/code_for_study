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

int selectRandom(vector<int>& data, int p ,int r, int k) {
  if(p == r) {
    return data[p];
  }
  int q = partition(data, p, r);
  int index = q - p + 1;
  if(k == index) {
    return data[q];
  } else if (k < index) {
    return selectRandom(data, p, index - 1, k);
  } else {
    return selectRandom(data, index + 1, r, k - index);
  }
}

int main() {
  vector<int> data {1,23,4,5,2,3,0};
  cout << selectRandom(data, 0, 5, 1) << endl;
  return 0;
}