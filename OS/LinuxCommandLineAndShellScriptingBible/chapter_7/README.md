# 7. Understanding Linux File Permissions

## 7.1 Linux Security

The core of the Linux security system is the *user account*. Each individual
who accesses a Linux system should have a unique user account assigned.

User permissions are tracked using a *user ID*, which is assigned to an
account when it's created.

### 7.1.1 The /etc/passwd file

The Linux system uses a special file to match the login name to
a corresponding UID value. This file is the `/etc/passwd` file.

The `root` user account is the administrator for the Linux system
and is always assigned UID 0. The Linux system creates lots
of user accounts for various functions that aren't actual users.
These are called *system accounts*. Linux reserves UIDs below 500 for system accounts.

The fields of the `/etc/passwd` file contain the following information:

+ The login username
+ The password for the user
+ The numerical UID of teh user account
+ The numerical group ID of the user account
+ A text description of the user account
+ The location of the `HOME` directory for the user
+ The default shell for the user

### 7.1.2 The /etc/shadow file

The `/etc/shadow` file provides more control over how the Linux system
manages passwords. Only the root user has access to the `/etc/shadow` file.

The `/etc/shadow` contains one record for each user account on the system.

There are nine fields in each `/etc/shadow` file record:

+ The login name corresponding to the login name in the `/etc/passwd` file.
+ The encrypted password
+ The number of days since January 1, 1970, that the password was last changed
+ The number of days before the password must be changed
+ The number of days before password expiration that the user
is warned to change the password
+ The number of days after a password expires before the account will be disabled
+ The date since the user account was disabled
+ A field reserved for the future use

### 7.1.3 Adding a new user

The primary tool used to add new users to your Linux system is `useradd`

### 7.1.4 Removing a user

If you want to remove a user from the system, the `userdel` command is what you need.

### 7.1.5 Modifying a user

Linux provides a few different utilities for modifying the information
for existing user accounts.

+ `usermod`: Edits user account fields
+ `passwd`: Changes the password for an existing user
+ `chpasswd`: Reads a file of login name and password pairs, and updates the password
+ `chage`: Changes the password's expiration date
+ `chfn`: Changes the user account's comment information
+ `chsh`: Changes the user account's default shell

## 7.2 Using Linux Groups

Group permissions allow multiple users to share a common set
of permissions for an object on the system.

Each group has a unique GID which is a unique numerical value
on the system. Along with the GID, each group has a unique group name.

### 7.2.1 The /etc/group file

The `/etc/group` file contains information about each group used on the system.

The `/etc/group` file uses four fields:

+ The group name
+ The group password
+ The GID
+ The list of user accounts that belong to the group

The group password allows a non-group member to temporarily become a member of the
group by using the password. This feature is not used all that commonly, but it does exist.

### 7.2.2 Creating new groups

The `groupadd` command allows you to create new groups on your system.

### 7.2.3 Modifying groups

The `groupmod` command allows you to change the GID or the group
name of an existing group.
