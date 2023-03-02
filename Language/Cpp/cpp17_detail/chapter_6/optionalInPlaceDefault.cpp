#include <iostream>
#include <optional>
#include <string>

class UserName {
private:
  std::string mName;

public:
  explicit UserName() : mName{"Default"} {
    std::cout << "UserName: UserName('";
    std::cout << mName << "')\n";
  }

  explicit UserName(const std::string &str) : mName(str) {
    std::cout << "UserName::UserName('";
    std::cout << mName << "')\n";
  }

  ~UserName() {
    std::cout << "UserName::~UserName('";
    std::cout << mName << "')\n";
  }

  UserName(const UserName &u) : mName(u.mName) {
    std::cout << "UserName::UserName(copy '";
    std::cout << mName << "')\n";
  }
  UserName(UserName &&u) noexcept : mName(std::move(u.mName)) {
    std::cout << "UserName::UserName(move '";
    std::cout << mName << "')\n";
  }
  UserName &operator=(const UserName &u) {
    mName = u.mName;
    std::cout << "UserName::=(copy '";
    std::cout << mName << "')\n";
    return *this;
  }
  UserName &operator=(UserName &&u) noexcept {
    mName = std::move(u.mName);
    std::cout << "UserName::=(move '";
    std::cout << mName << "')\n";
    return *this;
  }
};

int main() {
  // std::optional<UserName> opt(UserName{});
  std::optional<UserName> opt(std::in_place);
}
