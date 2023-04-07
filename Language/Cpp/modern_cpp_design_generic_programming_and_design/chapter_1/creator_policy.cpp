#include <iostream>

class Widget{};

template <typename T>
struct OpNewCreator {
  static T* Create() {
    return new T;
  }
};

template <typename T>
struct MallocCreator {
  static T* Create() {
    void *buf = std::malloc(sizeof(T));
    if (!buf) return 0;
    return new(buf) T;
  }
};

template <typename T>
class PrototypeCreator {
private:
  T* pPrototype_;

public:
  PrototypeCreator(T *pObj = nullptr): pPrototype_(pObj) {}
  T *Create() { return pPrototype_ ? pPrototype_->Clone() : 0; }
  T *GetPrototype() { return pPrototype_; }
  void SetPrototype(T *pObj) { pPrototype_ = pObj; }
};
