#include <iostream>
#include <vector>
using namespace std;

pair<int, int> ants(int length, vector<int>& distance) {
  int minTime = 0, maxTime = 0;
  for(int i = 0; i < distance.size(); ++i) {
    minTime = max(minTime, min(distance[i], length - distance[i]));
  }
  for(int i = 0; i < distance.size(); ++i) {
    maxTime = max(maxTime, max(distance[i], length - distance[i]));
  }
  return make_pair(minTime, maxTime);
}

int main() {
  int caseScale = 0;
  int length = 0, numberOfAnts = 0;
  vector<int> distance;
  distance.reserve(1000000);
  cin >> caseScale;
  for(int i = 0; i < caseScale; ++i) {
    cin >> length >> numberOfAnts;
    for(int j = 0; j < numberOfAnts; ++j) {
      int dis; cin >> dis;
      distance.push_back(dis);
    }
    pair<int, int> minMaxTime = ants(length, distance);
    cout << minMaxTime.first << " " << minMaxTime.second << "\n";
    distance.clear();
  }
  return 0;
}