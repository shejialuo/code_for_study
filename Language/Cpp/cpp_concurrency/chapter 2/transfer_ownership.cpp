/*
  * In join_thread.cpp, We use ThreadGuard to implement
  * the RAII, now we can reimplement this to use move
  * semantics
*/

#include <thread>
#include <iostream>
using namespace std;

/*
  * In the previous example, we use reference to
  * point to the out-scoped thread, it's a bad idea.
  * We should use move semantics.
*/

class Functor {
private:
  int& i;
public:
  Functor(int& i_):i(i_) {}
  void operator()(){
    cout << "Hello Thread\n";
  }

};

class ScopedThreadGuard {
private:
  thread t;
public:
  ScopedThreadGuard(thread t_) {
    t = move(t_);
  }
  ScopedThreadGuard(const ScopedThreadGuard&) = delete;
  ScopedThreadGuard& operator=(const ScopedThreadGuard&) = delete;
  ~ScopedThreadGuard() {
    if(t.joinable()) {
      t.join();
    }
  }
};

//! It's a good iead to use RAII
void joinThreadGoodIead() {
  int localState = 0;
  Functor func(localState);
  thread threadFunctor(func);
  //! Note that we use move here
  ScopedThreadGuard guard(move(threadFunctor));
}

int main() {
  joinThreadGoodIead();
  return 0;
}