object PatternMatch {
  abstract class Expr
  case class Var(name: String) extends Expr
  case class Number(name: Double) extends Expr
  case class UnOp(operator: String, arg: Expr) extends Expr
  case class BinOp(operator: String,
                  left: Expr, right: Expr) extends Expr

  /*
    * It adds a factory method with the name of the class
    * You can write Var("x") instead of new Var("x")
  */
  val v = Var("x")
  val op = BinOp("+", Number(1), v)
  /*
    * All arguments in the parameter list
    * implicitly get a val prefix
  */
  val vName = v.name

  /*
    * The compiler adds "natural" implementations
    * of methods toString, hashCode and equals
  */

  /*
    * The compiler adds a copy method to class
    * for making modified copy
  */

  val opCopy = op.copy(operator = "-")

  def simplifyTop(expr: Expr): Expr = expr match {
    case UnOp("-", UnOp("-", e)) => e
    case BinOp("+", e, Number(0)) => e
    case BinOp("*", e, Number(1)) => e
    case _ => expr
  }

  def simplifyAdd(expr: Expr): Expr = expr match {
    case BinOp("+", x, y) if x == y =>
      BinOp("*", x Number(2))
    case _ => expr
  }

}
