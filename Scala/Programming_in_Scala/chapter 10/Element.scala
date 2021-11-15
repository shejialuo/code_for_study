abstract class Element {
  def contents: Array[String]
  /*
    * It is encouraged in Scala to define methods that
    * take no parameter and have no side effects
    * as parameterless methods.
    * On the other hand, you should never define a method
    * that has side-effects without parentheses
  */
  def height: Int = contents.length
  def width: Int = if (height == 0) 0 else contents(0).length

}

class ArrayElement(
  val contents: Array[String]
) extends Element

class LineElement(s: String) extends ArrayElement(Array(s)) {
  override def width = s.length
  override def height = 1
}

class UniformElement (
  ch: Char,
  override val width: Int,
  override val height: Int
) extends Element{
  private val line = ch.toString * width
  def contents = Array.fill(height)(line)
}
