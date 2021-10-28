package main

import "fmt"

type user struct {
  name string
  email string
  ext int
  privileged bool
}

type admin struct {
  person user
  level string
}

func main() {
  var bill user
  fmt.Println(bill)

  lisa := user {
    name: "Lisa",
    email: "lisa@email.com",
    ext: 123,
    privileged: true,
  }
  fmt.Println(lisa)

  lisaAnotherForm := user{"Lisa", "lisa@email.com", 123, true}
  fmt.Println(lisaAnotherForm)

  fred := admin{
    person: user {
      name: "Lisa",
      email: "lisa@email.com",
      ext: 123,
      privileged: true,
    },
    level: "super",
  }
  fmt.Println(fred)
}
