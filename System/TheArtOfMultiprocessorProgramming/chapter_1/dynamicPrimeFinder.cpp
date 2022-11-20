#include <thread>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <array>

class Counter {
private:
  unsigned value;
  std::mutex lock ;
public:
  explicit Counter(unsigned i): value{i}, lock{} {}
  Counter(const Counter& counter) = delete;
  Counter& operator=(const Counter& counter) = delete;
  Counter(Counter&& counter) = delete;
  Counter& operator=(Counter&& counter) = delete;
  unsigned getAndIncrement() {
    std::lock_guard<std::mutex> guard{lock};
    value++;
    return value;
  }
};

bool isPrime(unsigned num) {
  for (unsigned i = 2; i < static_cast<unsigned>(std::sqrt(num)); ++i) {
    if (num % i == 0) {
      return false;
    }
  }
  return true;
}

void primePrint(unsigned threadID, Counter& counter) {
  unsigned i = 0;
  unsigned limit = static_cast<unsigned>(std::pow(10, 10));
  while (i < limit) {
    i = counter.getAndIncrement();
    if (isPrime(i)) {
       std::printf("%u is a prime\n", i);
    }
  }
}

int main() {
  std::array<std::thread, 10> threads{};
  Counter counter{1};
  for (int i = 0; i < 10; ++i) {
    threads[i] = std::move(std::thread(primePrint, i, std::ref(counter)));
  }
  for (int i = 0; i < 10; ++i) {
    threads[i].join();
  }
  return 0;
}
