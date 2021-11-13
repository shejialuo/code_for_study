import javax.management.relation.Relation
class Rational(n: Int, d: Int) {
  require(d != 0)

  private val g = gcd(n.abs, d.abs)
  val numer: Int = n / g
  val denom: Int = d / g
  def this(n: Int) = this(n, 1)
  override def toString = s"$n/$d"
  def + (rational: Rational): Rational =
    new Rational(
      number * rational.denom + rational.numer * denom,
      denom * rational.denom
    )

  def + (i: Int): Rational =
    new Rational(numer + i * denom, denom)

  def - (rational: Rational): Rational =
    new Rational(
      numer * rational.denom - rational.numer * denom,
      denom * rational.denom
    )

  def - (i: Int): Rational =
    new Rational(numer - i * denom, denom)

  def * (rational: Rational): Rational =
    new Rational(numer * rational.numer, denom * rational.denom)

  def * (i: Int): Rational =
    new Rational(numer * i , denom)

  def / (rational: Rational) =
    new Rational(numer * rational.denom, denom * rational.numer)

  def / (i : Int): Rational =
    new Rational(numer, denom * i)

  private def gcd(a: Int, b: Int): Int =
    if (b == 0) a else gcd(b, a % b)
}