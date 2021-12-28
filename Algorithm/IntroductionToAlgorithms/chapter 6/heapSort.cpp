#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

int left(int i) {
  return 2 * i;
}

int right(int i) {
  return 2 * i + 1;
}

void maxHeapify(vector<int>& data, int i , int heapSize) {
  int largest = i;
  while(true) {
    if(left(i) < heapSize && data[left(i)] > data[largest]) {
      largest = left(i);
    }
    if(right(i) < heapSize && data[right(i)] > data[largest]) {
      largest = right(i);
    }
    if(largest != i) {
      swap(data[i], data[largest]);
      i = largest;
    } else {
      return;
    }
  }
}

void buildMaxHeap(vector<int>& data) {
  for(int i = data.size() / 2; i >= 0; --i) {
    maxHeapify(data, i, data.size());
  }
}

void heapSort(vector<int>& data, int heapSize) {
  buildMaxHeap(data);
  for(int i = data.size() - 1; i > 0; --i) {
    swap(data[i], data[0]);
    heapSize--;
    maxHeapify(data, 0, heapSize);
  }
}

int main() {
  vector<int> data {1,3,2,1,5,1,2,3};
  heapSort(data, data.size());
  copy(data.cbegin(), data.cend(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  return 0;
}