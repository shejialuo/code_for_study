#include <memory>

class DirectAddressHash {
  std::unique_ptr<int> T;
  int keyNum;

  DirectAddressHash(const DirectAddressHash&) = delete;
  DirectAddressHash& operator=(const DirectAddressHash&) = delete;

  DirectAddressHash(int num): keyNum{num} {
    T = std::make_unique<int>(num);
  }

  int search(int key) {
    return T.get()[key];
  }

  void insert(int key, int num) {
    T.get()[key] = num;
  }

  void delete_(int key, int num) {
    T.get()[key] = 0;
  }
};
