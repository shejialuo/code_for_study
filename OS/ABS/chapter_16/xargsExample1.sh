#!/bin/bash

# A filter for feeding arguments to a command, and also
# a tool for assembling the commands themselves. It breaks
# a data stream into small enough chunks for filters and
# commands to process.

ls -l | xargs

find ~/Mail -type f | xargs grep "Linux"

# An interesting `xargs` option is `-n`, which limits the
# number of arguments passed.

ls | xargs -n 8 echo

# The `-P` option to `xargs` permits running processes in
# parallel

# Converts all the git images in current directory to png
# Options:
# =======
# -t  Print command to stderr.
# -n1 At most 1 argument per command line.
# -P2 Run up to 2 processes simultaneously

ls *gif | xargs -t -n1 -P2 git2png
