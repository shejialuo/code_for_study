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

class HtmlBuilder {
public:
  HtmlElement root;

  HtmlBuilder(string rootName) {
    root.name = rootName;
  }

  HtmlBuilder& addChild(string childName, string childText) {
    HtmlElement element{childName, childText};
    root.elements.emplace_back(element);
    return *this;
  }

  string str() { return root.str();}
};

int main() {
  HtmlBuilder ulBuilder{"ul"};
  ulBuilder.addChild("li", "hello").addChild("li", "world");
  cout << ulBuilder.str() << endl;
  return 0;
}
