#include <iostream>

constexpr bool doIsPrime(unsigned p, unsigned d) {
  return d != 2 ? (p % d != 0) && doIsPrime(p, d - 1)
                : (p % 2 != 0);
}

constexpr bool isPrime(unsigned p) {
  return p < 4 ? !(p < 2)
               : doIsPrime(p, p / 2);
}

int main() {
  if (isPrime(9)) {
    std::cout << "9 is the prime\n";
  } else {
    std::cout << "9 is not the prime\n";
  }
  return 0;
}
