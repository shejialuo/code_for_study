#include <fstream>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

/*
 * An imperative way
*/

vector<int> countLinesInFiles(const vector<string>& files) {
  vector<int> results;
  char c = 0;

  for(const auto& file: files) {
    int line_count = 0;
    ifstream in(file);
    while(in.get(c)) {
      if (c == '\n')
        line_count++;
    }

    results.push_back(line_count);
  }
  return results;
}
