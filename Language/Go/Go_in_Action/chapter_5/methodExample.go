package main

import "fmt"

type user struct {
	name  string
	email string
}

// notify implements a method with a value receiver.
func (u user) notify() {
	fmt.Printf("Sending User Email To %s<%s>\n", u.name, u.email)
}

// changeEmail implements a method with a pointer receiver
func (u *user) changeEmail(email string) {
	u.email = email
}

func main() {
	// Values of type user can be used to call methods
	// declared with a value receiver
	bill := user{"Bill", "bill@email.com"}
	bill.notify()

	// Pointers of type user can be used to call methods
	// declared with a value receiver.
	lisa := &user{"Lisa", "lisa@email.com"}
	lisa.notify()

	// Values of type user can be used to call methods
	// declared with a pointer receiver
	bill.changeEmail("bill@newdomain.com")
	bill.notify()

	// Pointers of type user can be used to call methods
	// declared with a pointer receiver.
	lisa.changeEmail("lisa@comcast.com")
	lisa.notify()
}
