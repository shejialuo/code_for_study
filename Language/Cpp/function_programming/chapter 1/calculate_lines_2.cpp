#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <string>
using namespace std;

/*
 * Still an imperative way, but use fewer states.
*/

int countLines(const string& filename) {
  ifstream in(filename);

  return count(istreambuf_iterator<char>(in),
               istreambuf_iterator<char>(),
               '\n');
}

vector<int> countLinesInFiles(const vector<string>& files) {
  vector<int> results;

  for(const auto& file: files) {
    results.push_back(countLines(file));
  }
  return results;
}
