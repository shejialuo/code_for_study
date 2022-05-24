#include <string>
#include <fstream>
#include <memory>
#include <cstdio>

/*
  * By default, the cleanup is a call of `delete`, assuming
  * that the object was created with `new`. But you can
  * define other ways to clean up objects. You can define your
  * own *destruction policy*. For example, if your object is
  * an array allocated with `new[]`, you have to define that
  * the cleanup performs a `delete[]`.
*/

class FileDeleter {
private:
  std::string filename;
public:
  FileDeleter(const std::string& fn): filename(fn) {}

  void operator()(std::ofstream* fp) {
    fp->close();
    std::remove(filename.c_str());
  }
};

int main() {
  std::shared_ptr<std::ofstream> fp(new std::ofstream("tmpfile.txt"),
                                    FileDeleter("tmpfile.txt"));

  return 0;
}