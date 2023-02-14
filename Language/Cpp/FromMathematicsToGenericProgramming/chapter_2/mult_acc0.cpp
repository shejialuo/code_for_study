bool odd(int n) { return n & 0x01; }
int half(int n) { return n >> 1; }

int mult_acc0(int r, int n, int a) {
  if (n == 1) return r + a;
  if (odd(n)) {
    return mult_acc0(r + a, half(n), a + a);
  } else {
    return mult_acc0(r, half(n), a + a);
  }
}
