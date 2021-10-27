#include <vector>
#include <numeric>
using namespace std;

double averageScore(const vector<int>& scores) {
  return accumulate(scores.cbegin(),scores.cend(), 0) / (double)scores.size();
}
