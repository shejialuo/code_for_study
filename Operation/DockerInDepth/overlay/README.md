# Overlay

You should read the original [doc](https://www.kernel.org/doc/Documentation/filesystems/overlayfs.txt).

## Upper and Lower

An overlay filesystem combines two filesystems:

+ Upper
+ Lower

When a name exists in both filesystems, the object in the "upper"
filesystem is visible when the "lower" filesystem is either hidden
or, in the case of directories, merged with the "upper" object.

The lower filesystem can be any filesystem supported by Linux
and does not need to be writable.

## Directories

Overlaying mainly involves directories. If a given name appears
in both upper and lower filesystems and refers to a non-directory
in either then the lower object is hidden. Where both upper and
lower objects are directories, a merged directory is formed.

At mount time, the two directories given as mount options `lowerdir` and
`upperdir` are combined into a merged directory:

```sh
mount -t overlay overlay -olowerdir=/lower,upperdir=/upper, workdir=/work /merged
```

That's the basic, it is enough for now.
