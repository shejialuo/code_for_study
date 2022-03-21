#!/bin/bash

# Removes all core dump files from user's home directory
find ~/ -name 'core*' -exec rm {} \;

# List all files in ~/projects directory tree
# that were modified within the last day
find ~/projects -mtime -1

# Same as above, but modified exactly one day ago
find ~/projects -mtime 1

# mtime = last modification time of the target file
# ctime = last status change time (via `chmod`)
# atime = last access time

DIR=/home/shejialuo/junk_files
# Curly brackets are placeholder for the path name by "find."

# Delete all files in "$DIR"
# that have not been accessed in at least 5 days
find "$DIR" -type f -atime +5 -exec rm {} ;\

# Finds all IPv4 addresses in /etc directory
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
