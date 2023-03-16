#include <variant>
#include <iostream>

class MyType {
public:
  MyType() { std::cout << "MyType::MyType\n"; }
  ~MyType() { std::cout << "MyType::~MyType\n";}
};

class OtherType {
public:
  OtherType() { std::cout << "OtherType::OtherType\n"; }
  ~OtherType() { std::cout << "OtherType::~OtherType\n"; }
};

int main() {
  std::variant<MyType, OtherType> v;
  v = OtherType();
  return 0;
}
