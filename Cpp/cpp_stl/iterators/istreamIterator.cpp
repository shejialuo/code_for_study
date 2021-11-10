#include <iostream>
#include <iterator>
using namespace std;

int main() {
  istream_iterator<int> intReader(cin);

  istream_iterator<int> intReaderEof;

  while(intReader != intReaderEof) {
    cout << *intReader << endl;
    ++intReader;
  }

  return 0;
}