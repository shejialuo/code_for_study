object Curry {
  def plainOldSum(x: Int, y: Int) = x + y
  def curriedSum(x: Int)(y: Int) = x + y
  val onePlus = curriedSum(1)_
}