#!/bin/bash

# Demonstrating some of the uses of 'expr'

echo

# Arithmetic Operators
# ---------- ---------

echo "Arithmetic Operators"
echo
a=`expr 5 + 3`
echo "5 + 3 = $a"
a=`expr $a + 1`
echo
echo "a + 1 = $a"
echo "(incrementing a variable)"

a=`expr 5 % 3`
echo
echo "5 mod 3 = $a"

echo
echo

# Logical Operators
# ------- ---------

# Returns 1 if true, 0 if false
# opposite of normal Bash convention.

echo "Logical Operators"
echo

x=24
y=25

b=`expr $x = $y`
echo "b = $b"
echo

a=3
b=`expr $a \> 10`
echo 'b=`expr $a \> 10, therefore...'
echo "If a > 10, b = 0 (false)"
echo "b = $b"
echo

b=`expr $a \< 10`
echo "If a < 10, b = 1 (true)"
echo "b = $b"

echo
echo

# String Operators
# ------ ---------

echo "String Operators"
echo

a=1234zipper43231
echo "The string being operated upon is \"$a\"."

b=`expr length $a`
echo "Length of \"$a\" is $b"

b=`expr index $a 23`
echo "Numerical position of first \"2\" in \"$a\" is \"$b\"."

b=`expr substr $a 2 6`
echo "Substring of \"$a\", starting at position 2,\
and 6 chars long is \"$b\"."
