#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <string>
using namespace std;

/*
 * An functional way
*/

int count_lines(const string& filename) {
  ifstream in(filename);

  return count(istreambuf_iterator<char>(in),
               istreambuf_iterator<char>(),
               '\n');
}

vector<int> count_lines_in_files(const vector<string>& files) {
  vector<int> results(files.size());

  transform(files.begin(), files.end(), results.begin(), count_lines);

  return results;
}
