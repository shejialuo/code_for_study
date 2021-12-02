/*
 * T is reference or pointer type
 * 1. If expr's type is a reference, ignore the reference part
 * 2. Then pattern-match expr's type against ParamType to determine T.
*/

template <typename T>
void f1(T& param) {}

void callByReferenceOrPointer() {
  int x = 27;
  const int cx = x;
  const int& rx = x;

  f1(x);         // T is int, param's type is int&
  f1(cx);        // T is const int, param's type is const int&
  f1(rx);        // T is const int, param's type is const int&
}


/*
 * Universal reference
 * 1. If expr is an lvalue, both T and ParamType are deduced to be
 *    lvalue references.
 * 2. IF expr is an rvalue, the normal rules apply
*/

template <typename T>
void f2(T&& param) {}

void callByUniversalReference() {
  int x = 27;
  const int cx = x;
  const int &rx = x;

  f2(x);         // T is int&, param's type is int&
  f2(cx);        // T is const int&, param's type is const int&
  f2(rx);        // T is const int&, param's type is const int&
  f2(27);        // T is int, param's type is int&&
}

/*
 * Neither a pointer nor a Reference
 * simple rule: decay!
*/

template <typename T>
void f3(T param) {}

void callByValue() {
  int x = 27;
  const int cx = x;
  const int &rx = x;

  f3(x);         // T is int, param is int
  f3(cx);        // T is int, param is int
  f3(rx);        // T is int, param is int
}
