bool odd(int n) { return n & 0x01; }
int half(int n) { return n >> 1; }

int mult_acc4(int r, int n, int a) {
  while (true) {
    if (odd(n)) {
      r = r + a;
      if (n == 1) return r;
    }
    n = half(n);
    a = a + a;
  }
}
