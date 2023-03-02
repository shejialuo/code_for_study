#include <iostream>
#include <optional>
#include <string>

class UserName {
private:
  std::string mName;

public:
  explicit UserName(std::string str) : mName{std::move(str)} {}
  ~UserName() { std::cout << "UserName::~UserName('" << mName << "')\n"; }
};

int main() {
  std::optional<UserName> oEmpty;

  oEmpty.emplace("Steve");

  oEmpty.emplace("Mark");

  oEmpty.reset();

  oEmpty.emplace("Fred");

  oEmpty = UserName("Joe");

  return 0;
}