#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

vector<pair<int, int> > lowHighPairVector;

void graveryardDesign(const int num) {
  int sum = 0;
  int pFirst = 0;
  int pLast = 0;
  while(pLast <= (int)sqrt(double(num))) {
    if (sum == num) {
      lowHighPairVector.push_back({pFirst, pLast});
    }
    if (sum > num) {
      sum += - (pFirst + 1) * (pFirst + 1);
      pFirst++;
      continue;
    }
    sum += (pLast + 1) * (pLast + 1);
    pLast++;
  }
}

int main() {
  int numberOfGraves;
  cin >> numberOfGraves;
  graveryardDesign(numberOfGraves);

  cout << lowHighPairVector.size() << "\n";

  for(int i = 0; i < lowHighPairVector.size(); ++i) {
    for(int j = 0 ; j < lowHighPairVector.size() - i - 1; ++j) {
      if(lowHighPairVector[j + 1].second - lowHighPairVector[j + 1].first >
         lowHighPairVector[j].second - lowHighPairVector[j].first) {
        swap(lowHighPairVector[j + 1], lowHighPairVector[j]);
      }
    }
  }

  for(auto& p: lowHighPairVector) {
    cout << p.second - p.first << " ";
    for(int i = p.first; i < p.second; ++i)
      cout << i + 1 << " ";
    cout << "\n";
  }
  return 0;
}