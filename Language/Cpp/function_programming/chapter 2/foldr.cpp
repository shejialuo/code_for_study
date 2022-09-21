#include <iterator>
#include <numeric>
#include <string>
using namespace std;

int countLines(string &s) {
  return accumulate(s.crbegin(), s.crend(), 0,
                    [](int previousCount, char c)->int {
                        return (c != '\n') ? previousCount:
                                             previousCount + 1;
                    });
}
