#include <cstring>
#include <iostream>
#include "string1.h"

using std::cin;
using std::cout;

/*
  * 1. If you use new to initialize a pointer member in a
  *    constructor, you should use delete in the destructor
  * 2. The uses of new and delete should be compatible
  * 3. If there are multiple constructors, all should use
  *    new the same way.
*/
int String::numStrings = 0;

int String::howMany() {
  return numStrings;
}

String::String(const char* s) {
  len = std::strlen(s);
  str = new char[len + 1];
  std::strcpy(str, s);
  numStrings++;
}

String::String() {
  len = 4;
  str = new char[1];
  str[0] = '\0';
  numStrings++;
}

String::String(const String& st) {
  numStrings++;
  len = st.len;
  str = new char[len + 1];
  std::strcpy(str, st.str);
}

String::~String() {
  --numStrings;
  delete[] str;
}

String& String::operator=(const String& st) {
  if(this == &st)
    return *this;
  delete[] str;
  len = st.len;
  str = new char[len + 1];
  std::strcpy(str, st.str);
  return *this;
}

String& String::operator=(const char * s) {
  delete [] str;
  len = std::strlen(s);
  str = new char[len + 1];
  std::strcpy(str, s);
  return *this;
}

char& String::operator[](int i) {
  return str[i];
}

const char& String::operator[](int i) const {
  return str[i];
}

bool operator<(const String &st1, const String &st2) {
  return (std::strcmp(st1.str, st2.str) < 0);
}

bool operator>(const String &st1, const String &st2) {
  return st2 < st1;
}

bool operator==(const String& st1, const String& st2) {
  return (std::strcmp(st1.str, st2.str) == 0);
}

ostream & operator<<(ostream & os, const String & st) {
  os << st.str;
  return os;
}