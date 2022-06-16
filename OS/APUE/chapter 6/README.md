# System Data Files and Information

## Login Accounting

Two data files provided with most UNIX systems are the `utmp` file,
which keeps track of all the users currently logged in, and the `wtmp` file,
which keeps track of all logins and logouts. With Version 7,
one type of record was written to both files, a binary record
consisting of the following structure:

```c++
struct utmp {
  char ut_line[8]; /* tty line */
  char ut_name[8]; /* login name */
  long ut_time;    /* seconds since Epoch */
};
```

On login, one of these structures was filled in and written to the
`utmp` file by the `login` program, and the same structure was
appended to the `wtmp` file. On logout, the entry in the `utmp`
file was erased filled with bull bytes by the `init` process,
and a new entry was appended to the `wtmp` file.

The `who` program reads the `utmp` file and prints its contents
in a readable form. Later versions of the UNIX system provides the `last` command,
which read through the `wtmp` file and printed selected entries.
