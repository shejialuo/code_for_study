#!/bin/bash


var1="a+b+c"
var2="d-e-f"
var3="g,h,i"

# The plus sign will be interpreted as a separator
IFS=+
echo $var1
echo $var2
echo $var3

IFS="-"
echo $var1
echo $var2
echo $var3

IFS=","
echo $var1
echo $var2
echo $vae3

echo

exit
