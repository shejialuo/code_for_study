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

int countLines(const string& filename) {
  ifstream in(filename);

  return count(istreambuf_iterator<char>(in),
               istreambuf_iterator<char>(),
               '\n');
}

vector<int> countLinesInFiles(const vector<string>& files) {
  vector<int> results(files.size());

  transform(files.begin(), files.end(), results.begin(), countLines);

  return results;
}
