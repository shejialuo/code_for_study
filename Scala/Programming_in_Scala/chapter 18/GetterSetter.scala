object GetterSetter {
  class Time {
    var hour = 12
    var minute = 0
  }

  class TimeExplicit {
    private[this] var h = 12
    private[this] var m = 0

    def hour: Int = h
    def hour_=(x: Int) = {h = x}

    def minute: Int = m
    def minute_=(x: Int) = {m = x}
  }

  class TimeExplicitRequire {
    private[this] var h = 12
    private[this] var m = 0

    def hour: Int = h
    def hour_=(x: Int) = {
      require(0 <= x && x < 24)
      h = x
    }

    def minute: Int = m
    def minute_=(x: Int) = {
      require(0 <= x && x< 60)
      m = x
    }
  }
  val time = new Time()
  val timeExplicit = new TimeExplicit()
  val timeExplicitRequire = new TimeExplicitRequire()
}