bool odd(int n) { return n & 0x01; }
int half(int n) { return n >> 1; }

int mult_acc2(int r, int n, int a) {
  if (odd(n)) {
    r = r +a;
    if (n == 1) return r;
  }
  return mult_acc2(r, half(n), a + a);
}
