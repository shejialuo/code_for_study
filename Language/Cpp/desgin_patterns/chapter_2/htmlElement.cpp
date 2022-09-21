#include <iostream>
#include <string>
#include <vector>
#include <cstdio>

using namespace std;

class HtmlElement {
public:
  string name;
  string text;
  vector<HtmlElement> elements;

  HtmlElement() {}
  HtmlElement(const string& name, const string& text)
    : name(name), text(text) {}
  string str(int indent = 0) const {
    // pretty-print the contents
  }
};

int main() {
  string words[] = {"hello", "world"};
  HtmlElement list{"ul", ""};
  for(auto w: words)
    list.elements.emplace_back(HtmlElement{"li", w});
  printf(list.str().c_str());
}
