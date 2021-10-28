package main

import "fmt"

func createSliceByMakeWithoutCapacity() {
  // Create a slice of strings
  // Contains a length and capacity of 5 elements.
  slice := make([]string, 5)
  fmt.Println(slice)
}

func createSliceByMakeWithCapacity() {
  slice := make([]int, 3, 5)
  fmt.Println(slice)
}

func createSliceByLiteral() {
  slice := []string{"Red", "Blue", "Green", "Yellow", "Pink"}
  fmt.Println(slice)
}

func createSliceByLiteralWithIndex() {
  // Create a slice of strings.
  // Initialize the 100th element with an empty string
  slice := []string{99: ""}
  fmt.Println(slice)
}

func differenceBetweenArraysAndSlices() {
  // Create an array
  array := [3]int {10,20,30}
  // Create a slice
  slice := []int {10,20,30}
  fmt.Println(array)
  fmt.Println(slice)
}

func declareNullSlice() {
  var slice []int
  fmt.Println(slice)
}

func basicOperationOnSlice() {

  // Change the value
  slice := []int{10, 20, 30, 40, 50}
  slice[1] = 25

  // Create a new slice
  // Contains a length of 2 and capacity of 4 elements
  newSlice := slice[1:3]
  fmt.Println(newSlice)

  // Change index 1 of newSlice
  // Change index 2 of the original slice.
  newSlice[1] = 35
  fmt.Println(newSlice[1])
  fmt.Println(slice[1])

  // Allocate a new element from capacity
  // Assign the value of 60 to the new element
  newSlice = append(newSlice, 60)
  fmt.Println(newSlice[2])
  fmt.Println(slice[3])

  source := []string{"Apple", "Orange", "Plum", "Banana", "Grape"}
  // Slice the third element and restrict the capacity
  // For slice[i:j:k]
  // Length: j - i
  // Capacity: k - i
  newSource := source[2:3:4]
  fmt.Println(newSource)
}

func copyIndependentSlice() {

  source := []string{"Apple", "Orange", "Plum", "Banana", "Grape"}

  // Slice the third element and restrict the capacity
  // Contains a length and capacity of 1 element
  slice := source[2:3:3]

  // Append a new string to the slice
  slice = append(slice,"Kiwi")
  fmt.Println(slice[1])
  fmt.Println(source[3])

}

func appendTwoSlice()  {
  s1 := []int{1, 2}
  s2 := []int{3, 4}
  fmt.Println( append(s1, s2...))
}

func main() {
  fmt.Println("This is the slice example.")
}

