#include "lock.hpp"
#include <unordered_map>
#include <thread>



class ThreadID {
public:
  static int get();
};

class LockOne: public Lock {
private:
  volatile bool flag[2] {};
public:
  virtual void lock() override {
    int i = ThreadID::get();
    int j = 1 - i;
    flag[i] = true;
    while (flag[j]) {}
  }
  virtual void unlock() override {
    int i = ThreadID::get();
    flag[i] = false;
  }
};
