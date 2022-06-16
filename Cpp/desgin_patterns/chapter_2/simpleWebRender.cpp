#include <string>
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;

int main() {
  string words[] = { "hello", "world" };
  ostringstream oss;
  oss << "<ul>";
  for(auto w: words) {
    oss << "  <li>" << w << "</li>";
  }
  oss << "</ul>";
  printf(oss.str().c_str());
  return 0;
}
