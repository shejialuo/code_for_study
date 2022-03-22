#!/bin/bash

empty()
{
}

exit 0 # Will not exit here!

func() {
  # Comment 1.
  # Comment 2.
  # This is still an empty function.
  # Thank you, for pointing this point out.
}
# Results in same error message as above.

not_empty() {
  :
} # Contains a : (null command), and this is okay.
