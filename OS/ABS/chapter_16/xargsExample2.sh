#!/bin/bash

LINES=5

echo ---------------------------------------------------------- >>logfile
tail -n $LINES /var/log/messages | xargs | fmt -s >> logfile
echo >>logfile
echo >>logfile

exit 0