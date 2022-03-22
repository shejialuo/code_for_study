#!/bin/bash

echo_var() {
  echo "$1"
}

message=Hello
Hello=Goodbye

echo_var "$message"
# Now, let's pass an indirect reference to the function.
echo_var "${!message}"

# What happens if we change the contents of "hello" variable?
Hello="Hello, again!"
echo_var "$message"        # Hello
echo_var "${!message}"     # Hello, again!

exit 0
