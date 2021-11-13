// Imperative way
var filename = "default.txt"
if (!args.isEmpty)
  filename = args(0)

// Functional way

filenameOther = if (!args.empty) args(0) else "default.txt"