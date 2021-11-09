package main

import (
  "fmt"
  "sync"
)

var memoryAcess sync.Mutex

var data int


func main() {
  go func() {
    memoryAcess.Lock()
    data++
    memoryAcess.Unlock()
  }()

  memoryAcess.Lock()
  if data == 0 {
    fmt.Printf("the value is 0.")
  } else {
    fmt.Printf("the value is %v.\n", data)
  }
  memoryAcess.Unlock()
}
