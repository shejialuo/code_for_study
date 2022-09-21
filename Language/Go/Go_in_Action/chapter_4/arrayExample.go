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

func defineArrayWithSpecificIndex() {
	array := [5]int{1: 10, 2: 20}
	fmt.Println(array)
}

func accessArrayWithIndex() {
	array := [5]int{10, 20, 30, 40, 50}
	array[2] = 35
}

func copyArrayWithOperatorEqual() {
	var array1 [5]string
	array2 := [5]string{"Red", "Blue", "Green", "Yellow", "Pink"}
	array1 = array2
	fmt.Println(array1)
}

func createTwoDimensionalArray() {
	array := [4][2]int{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
	fmt.Println(array)
}

func main() {
	fmt.Println("This is array example")
}
