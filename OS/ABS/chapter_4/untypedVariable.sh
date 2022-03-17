#!/bin/bash

a=2334
let "a += 1"
# a = 2335
echo "a = $a "
echo

# Substitute "BB" for 23
b=${a/23/BB}

# b = BB35
echo "b = $b"

declare -i b
echo "b = $b"

let "b += 1"    # BB35 + 1
echo "b = $b"   # b = 1
echo

c=BB34
echo "c = $c"
d=${c/BB/23} # This makes $d an integer

echo "d = $d"
let "d += 1"
echo "d = $d"
echo

# What about null variables?
e=''
echo "e = $e"
let "e += 1"   # e = 1
echo "e = $e"
echo

# What about undeclared variables?
echo "f = $f"
let "f += 1"   # f = 1
echo "f = $f"
echo

exit 0
