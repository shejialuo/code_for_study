#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>

using namespace std;

class HtmlBuilder;

class HtmlElement {
public:
  string name;
  string text;
  vector<HtmlElement> elements;
  const size_t indentSize = 2;

  static unique_ptr<HtmlBuilder> build(const string& rootName) {
    return make_unique<HtmlBuilder>(rootName);
  }

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
  auto builder = HtmlElement::build("ul");
  builder->addChild("li", "hello").addChild("li", "world");
  cout << builder->str() << endl;
  return 0;
}
