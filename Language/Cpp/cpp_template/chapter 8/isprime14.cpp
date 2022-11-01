#include <iostream>

constexpr bool isPrime(unsigned int p) {
  for (unsigned int d = 2; d <= p / 2; ++d) {
    if (p % d == 0) {
      return false;
    }
  }
  return p > 1;
}

int main() {
  if (isPrime(9)) {
    std::cout << "9 is the prime\n";
  } else {
    std::cout << "9 is not the prime\n";
  }
  return 0;
}
