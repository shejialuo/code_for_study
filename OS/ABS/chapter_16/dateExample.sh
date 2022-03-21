#!/bin/bash

# Needs a leading '+' to invoke formatting.
# %j gives day of year.
echo "The number of days since the year's beginning is `date +%j`."

echo "The number of seconds elapsed since 01/01/1970 is `date +%s`."

# It's great for creating "unique and random" temp filenames 
prefix=temp
suffix=$(date +%s)
filename=$prefix.$.suffix
echo "Temporary filename = $filename"

exit 0
