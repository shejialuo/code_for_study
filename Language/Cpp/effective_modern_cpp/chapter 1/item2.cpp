/*
 * When a variable is declared using auto, auto plays the role of
 * T in the template, and the type specifier for the variable
 * acts as ParamType.
*/

#include <iostream>

int main() {

  const char name[] = "R. N. Briggs";

  auto arr1 = name;        // arr1's type is const char*
  auto& arr2 = name;       // arr2's type is const char(&)[13]

  auto x1 = 27;
  auto x2(27);

  auto x3 = {27};          // type is initializer_list<int>
  auto x4{27};             // type is initializer_list<int>
}
