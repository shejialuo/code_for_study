#include <thread>
#include <iostream>
using namespace std;

void threadFunc() {
  cout << "Hello Thread" << "\n";
}

class ThreadClass {
private:
  int value;
public:
  ThreadClass(int val): value(val) {}
  // ThreadClass(ThreadClass &) = delete;
  void operator()() const {
    cout << this->value << "\n";
  }
};

int main() {
  thread threadByFunction(threadFunc);
  /*
    thread threadByFunctor(ThreadClass(5));
    will cause most vexing parsing. 
  */
  thread threadByFunctor{ThreadClass(5)};
  thread threadByLambda([](){
    std::cout << "This is lambda\n";
  });
  //! cout is not thread-safe.
  threadByFunction.detach();
  threadByFunctor.detach();
  threadByLambda.detach();
  return 0;
}