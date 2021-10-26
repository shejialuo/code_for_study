#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
using namespace std;

class LRU {
private:
  int length;
  vector<int> pageVector;
public:
  LRU(int buffer){
    pageVector.assign(buffer, -1);
    this->length = buffer;
  }
  void LRU_Algorithm(int commingPage) {

    int index = -1;
    for(int i = 0; i < length; ++i) {
      if(pageVector[i] == commingPage) {
        index = i;
        break;
      }
    }

    // not in the pageVector
    if(index == -1) 
      index = length - 1;
    
    pageVector[0] = commingPage;
  }
};
