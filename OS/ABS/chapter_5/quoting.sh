#!/bin/bash

List="one two three"

for a in $List
do
  echo "$a"
done

# one
# two
# three

echo "---"

for a in "$List" # Preserves whitespace in a single variable.
do
  echo "$a"
done

# one two three

exit 0
