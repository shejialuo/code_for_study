#include <iostream>
#include <string>
#include <variant>

using namespace std;

struct PrintVisitor {
  void operator()(int i) { cout << "int: " << i << '\n'; }
  void operator()(float f) { cout << "float: " << f << '\n'; }
  void operator()(const string &s) { cout << "str: " << s << '\n'; }
};

int main() {
  variant<int, float, string> intFloatString;
  static_assert(variant_size_v<decltype(intFloatString)> == 3);

  // Use `visit` to call the corresponding type
  visit(PrintVisitor{}, intFloatString);

  // index will show the current used 'type'
  cout << "index = " << intFloatString.index() << endl;
  intFloatString = 100.0f;
  cout << "index = " << intFloatString.index() << endl;
  intFloatString = "hello super world";
  cout << "index = " << intFloatString.index() << endl;

  // try with get_if
  if (const auto intPtr = get_if<int>(&intFloatString)) {
    cout << "int: " << *intPtr << '\n';
  } else if (const auto floatPtr = get_if<float>(&intFloatString)) {
    cout << "float: " << *floatPtr << '\n';
  }

  if (holds_alternative<int>(intFloatString)) {
    cout << "the variant holds an int\n";
  } else if (holds_alternative<float>(intFloatString)) {
    cout << "the variant holds a float\n";
  } else if (holds_alternative<string>(intFloatString)) {
    cout << "the variant holds a string\n";
  }

  try {
    auto f = get<float>(intFloatString);
    cout << "float! " << f << '\n';
  } catch (bad_variant_access&) {
    cout << "our variant doesn't hold float at this momenet...\n";
  }

  return 0;

}
