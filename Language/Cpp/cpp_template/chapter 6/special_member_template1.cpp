#include <utility>
#include <string>
#include <iostream>

class Person {
private:
  std::string name;
public:
  explicit Person(std::string const& n) : name(n) {
    std::cout << "copying string-CONSTR for '" << name << "'\n";
  }
  explicit Person(std::string&& n) : name(std::move(n)) {
    std::cout << "moving string-CONSTR for '" << name << "'\n";
  }
  Person(Person const& p) : name(p.name) {
    std::cout << "COPY-CONSTR Person '" << name << "'\n";
  }
  Person(Person&& p) : name(std::move(p.name)) {
    std::cout << "MOVE-CONSTR person '" << name << "'\n";
  }
};

int main() {
  std::string s = "sname";
  Person p1{s};
  Person p2{"tmp"};
  Person p3{p1};
  Person p4{std::move(p1)};
}
