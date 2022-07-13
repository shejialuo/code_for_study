package main

import "fmt"

func main() {
	stringStream := make(chan string)
	close(stringStream)
	go func() {
		stringStream <- "Hello channels"
	}()
	salutation, ok := <-stringStream
	fmt.Printf("(%v): %v\n", ok, salutation)
}
