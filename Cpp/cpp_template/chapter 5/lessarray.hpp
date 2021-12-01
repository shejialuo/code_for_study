#ifndef LESSARRAY_HPP
#define LESSARRAY_HPP

/*
 * When passing raw arrays or string iterals to templates, some care
 * has to be taken. First, if the template parameters are declared
 * as references, the arguments don't decay. That is, a passed
 * argument of "hello" has type const char[6]. This can be a problem.
*/

template <typename T, int N, int M>
bool less(T(&a)[N], T(&b)[M])  {
  for(int i = 0; i < N && i < M; ++i) {
    if (a[i] < b[i]) return true;
    if (b[i] < a[i]) return false;
  }
}

#endif // LESSARRAY_HPP
