#!/bin/bash

# The - option to a here document <<-
# suppresses leading tabs in the body of the document,
# but not spaces

cat <<-ENDOFMESSAGE
				This is line 1 of the message
				This is line 2 of the message
				This is line 3 of the message
ENDOFMESSAGE

exit 0
