#!/bin/bash

# Try invoking this script with:
# '-mn'
# '-oq qOption' (qOption can be some arbitrary string)
# '-qXXX -r'
# '-qr' Unexpected result
# '-q -r' Unexpected result
# '-mnop -mnop' Unexpected result
# If an option expects an argument, then it will grab


NO_ARGS=0
E_OPTERROR=85

if [ $# -eq "$NO_ARGS" ]
then
  echo "Usage: `basename $0` options (-mnopqrs)"
  exit $E_OPTERROR
fi


while getopts ":mnopq:rs" Option
do
  case $Option in
    m      ) echo "Scenario #1: option -m-  [OPTIND=${OPTIND}]";;
    n  | o ) echo "Scenario #2: option -$Option- [OPTIND=${OPTIND}]";;
    p      ) echo "Scenario #3: option -p- [OPTIND=${OPTIND}]";;
    q      ) echo "Scenario #4: option -q- with argument \"$OPTARG\" [OPTIND=${OPTIND}]";;
    r  | s ) echo "Scenario #5: option -$Option-";;
    *      ) echo "Unimplemented option choses.";;
  esac
done

shift $(($OPTIND - 1))

exit $?
