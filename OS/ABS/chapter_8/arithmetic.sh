#!/bin/nash

n=1; echo -n "$n "

let "n = $n + 1"  # let "n = n + 1" also works.
echo -n "$n "

# ":" necessary because otherwise Bash attempts
# to interpret $((n =$n + 1)) as a command
: $((n =$n + 1))

(( n = n + 1 ))
echo -n "$n "

let "n++"
echo -n "$n "

(( n++ ))
echo -n "$n "

echo

exit 0
