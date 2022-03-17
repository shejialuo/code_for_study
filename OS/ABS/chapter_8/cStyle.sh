#!/bin/bash

echo

(( a = 23)) # Setting a value, C-Style

echo "a (initial value) = $a"

(( a++ ))
echo "a (after a++) = $a"

(( a-- ))
echo "a (after a--) = $a"

echo

exit
