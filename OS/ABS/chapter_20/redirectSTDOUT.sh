#!/bin/bash

LOGFILE=logfile.txt

# Link file descriptor #6
exec 6>&1

exec > $LOGFILE

# All output form commands in this block sent to file $LOGFILE.

echo -n "Logfile: "
date
echo "---------------------------"
echo

echo "Output of \"ls -al\" command"
echo
ls -al
echo; echo
echo "Output of \"df\" command"
echo
df

# Restore stdout and close file descriptor #6
exec 1>&6 6>$-

exit 0
