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

int count_lines(const string& filename) {
  ifstream in(filename);

  return count(istreambuf_iterator<char>(in),
               istreambuf_iterator<char>(),
               '\n');
}

vector<int> count_lines_in_files(const vector<string>& files) {
  vector<int> results;

  for(const auto& file: files) {
    results.push_back(count_lines(file));
  }
  return results;
}
