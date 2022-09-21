object ClientCode {
  // Imperative way
  def containsNeg(nums: List[Int]): Boolean = {
    val exists = false
    for (num <- nums)
      if (num < 0)
        exists = true
    exists
  }

  // Functional way
  def containsOdd(nums: List[int]) = nums.exists(_ % 2 == 1)
}