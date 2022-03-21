# I/O Redirection

```sh
COMMAND_OUTPUT >
  # Redirect stdout to a file.
  # Creates the file if not present, otherwise overwrites it.

  ls -lR > dir-tree.list

: > filename
  # The > truncates file "filename" to zero length.
  # If file not present, creates zero-length file
  # The : serves as a dummy placeholder, producing no output.

> filename
  # The > truncates file "filename" to zero length.
  # If file not present, creates zero-length file.

COMMAND_OUTPUT >>
  # Redirect stdout to a file.
  # Creates the file if not present, otherwise appends to it.
  # Single-line redirection commands (affect only the line they are on):

1 > filename
  # Redirect stdout to file "filename"
& > filename
  # Redirect both stdout and stderr to file "filename"
M > N
  # "M" is a file descriptor, which defaults to 1, if not explicitly set.
  # "N" is a filename.
  # File descriptor "M" is redirect to file "N".

N >&N
  # "M" is a file descriptor
  # "N" is another file descriptor

[j]<>filename
  # Open file "filename" for reading and writing,
  # and assign file descriptor "j" to it.
  # If "filename" does not exist, create it
  # If file descriptor "j" is not specified,
  # default to fd 0
n <&-
  # Close input file descriptor n
n >&-
  # Close output file descriptor n
```
