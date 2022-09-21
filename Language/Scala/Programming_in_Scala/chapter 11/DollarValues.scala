class Dollars(val amount: Int) extends AnyVal {
  override def toString() = "$" + amount
}

/*
  Test:
  scala> val money = new Dollars(1000000)
  scala > money.amount
*/