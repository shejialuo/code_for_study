import scala.collection.mutable.ListBuffer
object SequencesExample {
  val colors = List("red", "blue", "green")
  val fiveInts = new Array[Int](5)
  val fiveToOne = Array(5, 4, 3, 2 ,1)
  val buf = new ListBuffer[Int]
  buf += 1
  buf += 2
  3 +=: buf
  val bufList = buf.toList
}