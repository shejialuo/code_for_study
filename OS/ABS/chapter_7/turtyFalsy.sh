#!/bin/bash

(( 0 && 1))
echo $? # 1

let "num = (( 0 && 1 ))"
echo $num # 0

let "num = (( 0 && 1 ))"
echo $? # 1

(( 200 || 11 ))
echo $? # 0

let "num = (( 200 || 11 ))"
echo $num # 1

let "num = (( 200 || 11 ))"
echo $? # 0

echo

echo "Testing \"0\""
if [ 0 ]
then
  echo "0 is true."
else
  echo "0 is false."
fi

echo

echo "Testing \"1\""
if [ 1 ]
then
  echo "1 is true."
else
  echo "1 is false."
fi

echo "Testing \"-1\""
if [ -1 ]
then
  echo "-1 is true."
else
  echo "-1 is false."
fi

echo

echo "Testing \"NULL\""
if [ ]
then
  echo "NULL is true"
else
  echo "NULL is false."
fi

echo

echo "Testing \"xyz\""
if [ xyz ]
then
  echo "Random string is true."
else
  echo "Random string is false."
fi

echo "Testing \"\$xyz\""
if [ $xyz ]
then
  echo "Uninitialized variable is true."
else
  echo "Uninitialized variable is false."
fi

echo

echo "Testing \"-n \$xyz\""
if [ -n "$xyz" ]
then
  echo "Uninitialized variable is true."
else
  echo "Uninitialized variable is false."
fi

exit 0
