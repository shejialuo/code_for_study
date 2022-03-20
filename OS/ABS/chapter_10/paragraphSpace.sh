#!/bin/bash

# Inserts a blank line between paragraphs of a single-spaced text file

MINLEN=60

while read line
do
  echo "$line"
  len=${#line}

  if [[ "$len" -lt "$MINLEN" && "$line" =~ [*{\.}]$ ]]; then 
    echo
  fi
done

exit
