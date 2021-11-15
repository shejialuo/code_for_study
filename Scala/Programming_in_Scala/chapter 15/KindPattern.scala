object KindPattern {
  def wildcardPattern(x: Any) = x match {
    case 1 => println("This is 1")
    case _ => println("It's something else")
  }

  def constantPatterns(x: Any) = x match {
    case 5 => "five"
    case true => "truth"
    case "hello" => "hi"
    case Nil => "hi!"
    case _ => "something else"
  }

  /*
    * a simple names starting with a lowercase
    * letter is taken to be a pattern variable;
    * all other references are taken to be constants.
  */
  val auxVariable = 3
  def variablePatterns(x: Any) = x match {
    case 0 => "zero"
    case auxVariable => "not zero" // As wildcardPattern
  }

  //! Bad style I think
  def variableAsConstantPatterns(x: any) = x match {
    case 0 => "zero"
    case `auxVariable` => "not zero"
    case _ => "Not a number" // Must use wildcard
  }

  def sequencePatterns(x: Any) = x match {
    case List(0,_,_) => println("found it")
    case _ => Unit
  }

  def typedPattern(x: Any) = x match {
    case s: String => s.length
    case m: Map[_, _] => m.size
    case _ => -1
  }

}