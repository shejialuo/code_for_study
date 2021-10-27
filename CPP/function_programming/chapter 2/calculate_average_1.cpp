#include <vector>
#include <algorithm>
using namespace std;

double averageScore(const vector<int>& scores) {
  int sum = 0;

  for(int score: scores) {
    sum += score;
  }

  return sum / (double)scores.size();
}
