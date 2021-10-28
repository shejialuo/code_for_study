package main

import "fmt"

func defaultArray() {
  var array [5]int
  fmt.Println(array)
}

func defineArrayWithLiteral() {
  array := [5]int{10, 20, 30, 40, 50}
  fmt.Println(array)
}

func defineArrayWithGoCalculatingSize() {
  array := [...]int{10, 20, 30, 40, 50}
  fmt.Println(array)
}

func main() {
  fmt.Println("This is array example")
}
