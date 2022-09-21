import java.io.File
import java.io.PrintWriter
import java.io.Writer
object ControlStructure {
  def withPrintWriter(file: File)(op: PrintWriter => Unit) = {
    val writer = new PrintWriter(file)
    try {
      op(writer)
    }finally {
      writer.close()
    }
  }

  withPrintWriter(new File("data.txt")){
    writer => writer.println(new java.util.Date)
}
}