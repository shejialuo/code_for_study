#!/bin/bash

# In contrast to C, a Bash variable declared inside a function
# is local ONLY if declared as such.

func() {
  local loc_var=23
  echo
  echo "\"loc_var\" in function = $loc_var"
  global_var=999
  echo "\"global_var\" in function = $global_var"
}

echo
echo "\"loc_var\" outside function = $loc_var"
echo "\"global_var\" outside function = $global_var"

echo

exit 0
