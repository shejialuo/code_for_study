#include <any>
#include <string>

int main() {
  std::any a;

  std::any a2{10};
  std::any a3{std::in_place_type<std::string>, "Hello World"};

  std::any a4 = std::make_any<std::string>("Hello World");
}
