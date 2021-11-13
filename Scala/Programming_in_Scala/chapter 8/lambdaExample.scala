val increase = (x: Int) => x + 1

println(increase (10))

val f = (_ : Int) + (_ : Int)

def sum(a: Int, b: Int, c: Int) = a + b + c

val partial1 = sum _
println(partial1(1,2,3))

val partial2 = sum(1, _: Int, 3)
println(partial2(2))


