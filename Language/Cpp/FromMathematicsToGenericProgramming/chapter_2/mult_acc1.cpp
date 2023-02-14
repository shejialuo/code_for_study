bool odd(int n) { return n & 0x01; }
int half(int n) { return n >> 1; }

int mult_acc1(int r, int n, int a) {
  if (n == 1) return r + a;
  if (odd(n)) {
    r = r +a;
  }
  return mult_acc1(r, half(n), a + a);
}
