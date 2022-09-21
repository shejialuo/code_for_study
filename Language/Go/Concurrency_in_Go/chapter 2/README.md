# 2. Modeling Your Code: Communicating Sequential Processes

## 2.1 What Is CSP?

CSP stands for "Communicating Sequential Processes" which is both a
technique and the name of the paper that introduced it. In
1978, Charles Antony Richard Hoare published the paper in ACM.

In this paper, Hoare suggests that input and output are two overlooked
primitives of programming—particularly in concurrent code.

To support his assertion that inputs and outputs needed to be
considered language primitives. Hoare's CSP programming language
contained primitives to model input and output, or *communication*,
between *processes* correctly. Hoare applied the term *processes* to any
encapsulated portion of logic that the required input run and produced
output other processes would consume.

For communication between the processes, Hoare created input and
output *commands*: `!` for sending input into a process, and `?` for
reading output from a process. Go is one of the first languages to
incorporate principles from CSP in its core, and bring the style
of concurrent programming to the masses.

## 2.2 Go's Philosophy on Concurrency

CSP was and is a large part of what Go was designed around; however,
Go also supports more traditional means of writing concurrent code
through memory access synchronization and the primitives that
follow that technique.

Go's philosophy on concurrency can be summed up like this: aim for
simplicity, use channels when possible, and treat goroutines like a free resource.
