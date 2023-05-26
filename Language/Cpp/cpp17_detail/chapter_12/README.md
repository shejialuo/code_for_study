# Chapter 12 Filesystem

With C++17, we've taken a big step forward in the form of the
`std::filesystem` component.

## Filesystem overview

Below we can define the core elements of this module:

+ The `std::filesystem::path` object allows you to manipulate paths that
represent existing or not existing files and directories in the system.
+ `std::filesystem::directory_entry` represents an existing path with
additional status information like last write time, file size, or other
attributes.
+ Directory iterators allow you to iterate through a given directory. The
library provides a recursive and non-recursive version.

## Demo

[filesystem_list_files.cpp](./filesystem_list_files.cpp)

## The Path Object

The core part of the library is the `path` object. It contains a pathname.
The object doesn't have to point to an existing file in the filesystem. The
path might even be in an invalid form.

The path is composed of the following elements:

```txt
root-name root-directory relative-path
```

+ `root-name`: POSIX systems don't have a root name. On Windows, it's usually
the name of a drive.
+ `root-directory`: distinguishes relative path from the absolute path
+ `relative-path`:
  + filename
  + directory separator
  + relative-path
