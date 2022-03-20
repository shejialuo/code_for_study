#!/bin/bash

# Notice the escaped spaces
set a\ b c d\ e;

OIFS=$IFS; IFS=:

echo

until [ $# -eq 0 ]
do
  echo "### k0 = "$k""
  # Append each pos param to loop variable
  k=$1:$k;
  echo "### k = "$k""
  echo
  shift
done

set $k
echo -
echo $#
echo

for i
do
  echo $i
done

IFS=$OIFS

exit 0
