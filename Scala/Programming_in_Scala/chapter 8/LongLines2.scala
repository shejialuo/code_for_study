import scala.io.Source

object LongLines {
  def processFile(filename: String, width: Int) = {
    def processLine(filename: String, width: Int, line: String) = {
      if(line.length > width)
        println(filename + ": " + line.trim)
    }

    val source = Source.fromFile(filename)
    for(line <- source.getLines()) {
      processLine(filename, width, line)
    }
  }
}