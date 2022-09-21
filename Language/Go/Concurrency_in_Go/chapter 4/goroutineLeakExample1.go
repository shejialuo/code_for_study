package main

import "fmt"

func main() {
	doWork := func(strings <-chan string) <-chan interface{} {
		completed := make(chan interface{})
		go func() {
			defer fmt.Println("doWork exited")
			defer close(completed)
			for s := range strings {
				fmt.Println(s)
			}
		}()
		return completed
	}
	// Here we passes a `nil` channel into `doWork`. Therefore,
	// the `strings` channel will never actually gets any strings
	// written onto it, and the goroutine containing `doWork` will
	// remain in the memory for the lifetime of this process.
	doWork(nil)
	fmt.Println("Done.")
}
