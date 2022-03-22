#!/bin/bash

f1() {
  f2() {
    echo "Function \"f2\", inside \"f1\"."
  }
}

f2 # Gives an error message.
   # Even a preceding "declare -f f2" wouldn't help.

echo

f1 # Does nothing, since calling "f1" does not automatically call "f2"
f2 # Now, it's all right to call "f2",
   # since its definition has been made visible by calling "f1".
