/*
  * When joining, it's important that a thread can be join.
  * You could use try-catch, but this is bad.
  * So we should use RAII
*/

#include <thread>
#include <iostream>
using namespace std;

class Functor {
private:
  int& i;
public:
  Functor(int& i_):i(i_) {}
  void operator()(){
    cout << "Hello Thread\n";
  }

};

class ThreadGuard {
private:
  thread& t;
public:
  explicit ThreadGuard(thread& t_): t(t_) {}
  ~ThreadGuard() {
    if(t.joinable()) {
      t.join();
    }
  }
  ThreadGuard(const ThreadGuard&) = delete;
  ThreadGuard& operator=(const ThreadGuard &) = delete;
};

//! It's a bad iead.
void joinThreadBadIdea() {
  int localState = 0;
  Functor func(localState);
  thread threadFunctor(func);
  try {
    cout << "This is a bad idea to use try-catch\n";
  }
  catch(...) {
    threadFunctor.join();
    throw;
  }
  threadFunctor.join();
}

//! It's a good iead to use RAII
void joinThreadGoodIdea() {
  int localState = 0;
  Functor func(localState);
  thread threadFunctor(func);
  ThreadGuard guard(threadFunctor);
}

int main() {
  joinThreadBadIdea();
  joinThreadGoodIdea();
  return 0;
}
