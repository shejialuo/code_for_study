/*
  * Just like Haskell
*/
object OptionType {
  val capitals = Map("France" -> "Paris", "Japan" -> "Tokyo")

  def show(x: Option[String]) = x match {
    case Some(s) => s
    case None => "?"
  }
}