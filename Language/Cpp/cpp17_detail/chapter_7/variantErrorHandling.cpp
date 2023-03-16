#include <iostream>
#include <string>
#include <variant>
enum class ErrorCode {
  Ok,
  SystemError,
  IoError,
  NetworkError
};

std::variant<std::string, ErrorCode> fetchNameFromNetwork(int i) {
  if (i == 0) {
    return ErrorCode::SystemError;
  }
  if (i == 1) {
    return ErrorCode::NetworkError;
  }
  return std::string{"Hello World!"};
}

int main() {
  auto response = fetchNameFromNetwork(0);
  if (std::holds_alternative<std::string>(response)) {
    std::cout << std::get<std::string>(response);
  } else {
    std::cout << "Error!\n";
  }
}
