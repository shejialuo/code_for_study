// Imperative way

def gcdLoop(x: Long, y: Long): Long = {
  var a = x
  var b = y
  while (a != 0) {
    var temp = a
    a = b % a
    b = temp
  }
  b
}

// Functional way
def gcd(x: Long, y: Long): Long =
  if (y == 0) x else gcd(y, x % y)