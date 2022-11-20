#include <thread>
#include <cmath>
#include <cstdio>
#include <array>

bool isPrime(unsigned num) {
  for (unsigned i = 2; i < static_cast<unsigned>(std::sqrt(num)); ++i) {
    if (num % i == 0) {
      return false;
    }
  }
  return true;
}

void primePrint(unsigned threadID) {
  unsigned block = static_cast<unsigned>(std::pow(10, 9));
  for (unsigned i = (threadID * block) + 1; i <= (threadID + 1) * block; ++i) {
    if (isPrime(i)) {
      std::printf("%u is a prime\n", i);
    }
  }
}

int main() {
  std::array<std::thread, 10> threads{};
  for (int i = 0; i < 10; ++i) {
    threads[i] = std::move(std::thread(primePrint, i));
  }
  for (int i = 0; i < 10; ++i) {
    threads[i].join();
  }
  return 0;
}
