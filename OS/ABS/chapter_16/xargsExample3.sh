#!/bin/bash

# Copy (verbose) all files in current directory
# to directory specified on command-line

E_NOARGS=85

if [[ -z "$1" ]]
then
  echo "Usage: `basename $0` directory-to-copy-to"
  exit $E_NOARGS
fi

# -i is "replace strings" option.
# {} is a placeholder option.

ls . | xargs -i -t cp ./{} $1
