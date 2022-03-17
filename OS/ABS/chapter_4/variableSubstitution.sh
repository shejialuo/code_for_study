#!/bin/bash

a=375
hello=$a

echo hello

echo $hello
echo ${hello}

echo "$hello"
echo "${hello}"

echo

hello="A B  C   D"
echo $hello   # A B C D
# Quoting a variable preserves whitespace
echo "$hello" # A B  C   D

echo

echo '$hello' # $hello

hello=
echo "\$hello (null value) = $hello"

# Uninitialized variable has null value
echo "uninitialized_variable = $uninitialized_variable"

uninitialized_variable=
# It still has a null value
echo "uninitialized_variable = $uninitialized_variable"

uninitialized_variable=23
unset uninitialized_variable
# It still has a null value
echo "uninitialized_variable = $uninitialized_variable"

echo

exit 0
