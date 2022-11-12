#include <iostream>

template<typename T>
void tmplParamIsReference(T) {
  std::cout << "T is reference: " << std::is_reference_v<T> << '\n';
}

int main() {
  std::cout << std::boolalpha;
  int i;
  int &r = i;
  tmplParamIsReference(i);
  tmplParamIsReference(r);
  tmplParamIsReference<int&>(i);
  tmplParamIsReference<int&>(r);
  return 0;
}
