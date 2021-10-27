#include <string>
#include <numeric>
#include <iostream>
using namespace std;

int countLines(const string& s) {
  return accumulate(s.cbegin(), s.cend(), 0,
                   [](int previousCount, char c) -> int {
                      return (c != '\n') ? previousCount
                                         : previousCount + 1;
                   });
}

