#include <iostream>
#include <string>
#include <utility>
#include <type_traits>

// C++17
template<typename T>
using EnableIfString_17 = std::enable_if_t<std::is_convertible_v<T, std::string>>;

// C++14
template<typename T>
using EnableIfString_14 = std::enable_if_t<std::is_convertible<T, std::string>::value>;

// C++11
template<typename T>
using EnableIfString_11 = typename std::enable_if<std::is_convertible<T, std::string>::value>::type;

class Person {
private:
  std::string name;
public:
  template<typename STR, typename = EnableIfString_17<STR>>
  explicit Person(STR&& n) : name{std::forward<STR>(n)} {
    std::cout << "TMPL-CONSTR for '" << name << "'\n";
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
