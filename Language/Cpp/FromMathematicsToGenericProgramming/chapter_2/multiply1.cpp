bool odd(int n) { return n & 0x01; }
int half(int n) { return n >> 1; }

int multiply1(int n, int a) {
  if (n == 1) return a;
  int result = multiply1(half(n), a + a);
  if (odd(n)) result += a;
  return result;
}
