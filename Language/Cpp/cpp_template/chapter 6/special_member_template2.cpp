#include <utility>
#include <string>
#include <iostream>

class Person {
private:
  std::string name;
public:
  template<typename STR>
  explicit Person(STR&& n) : name(std::forward<STR>(n)) {
    std::cout << "TMPL-CONSTR for '" << name << "'\n";
  }
  Person(Person const& p) : name(p.name) {
    std::cout << "COPY-CONSTR Person '" << name << "'\n";
  }
  Person(Person&& p) : name(std::move(p.name)) {
    std::cout << "MOVE-CONSTR person '" << name << "'\n";
  }
};
