#include <string>
#include <algorithm>
#include <iostream>
using namespace std;

bool isNotSpace(char c) {
  if(c != ' ')
    return true;
  return false;
}

string trimLeft(string s) {
  s.erase(s.begin(),find_if(s.begin(), s.end(), isNotSpace));
  return s;
}

string trimRight(string s) {
  s.erase(find_if(s.rbegin(), s.rend(), isNotSpace).base(), s.end());
  return s;
}

string trim(string s) {
  return trimLeft(trimRight(move(s)));
}

