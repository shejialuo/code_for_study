# Cabal User Guide

## 1. Introduction

Cabal is a package system for Haskell software. Packaging systems deal with packages
and with Cabal we call them *Cabal packages*. The Cabal package is the unit of
distribution.

When distributed, Cabal packages use the standard compressed format, with the
file extension `.tar.gz`.

### 1.1 A tool for working with packages

There is a command line tool, called `cabal`, that users and developers can use
to build and install Cabal packages. It can be used for both local packages
and for packages available remotely over the network. It can automatically
install Cabal packages plus any other Cabal packages they depend on.

```sh
cd cabalProject/
cabal install
```

It is also possible to install several local packages at once.

```sh
cabal install cabalProject/package1/
```

### 1.2 What's in a package

A Cabal package consists of:

+ Haskell software
+ Including libraries
+ Executables
+ Tests
+ Metadata about the package (`.cabal`)
+ A standard interface to build the package (`Setup.hs`)

## 2. Package Concepts and Development

### 2.1 Quick Start

Let' assume we have created a project directory and already have a Haskell module
or two.

Every project needs a name, we'll call this example `proglet`

To turn this into a Cabal package we need two extra files in the project's
root directory:

+ `proglet.cabal`: containing package metadata and build information.
+ `Setup.hs`: usually containing a few standardized lines of code.

We can create both files manually or we can use `cabal init`.

#### 2.1.1 Modules included in the package

For a library, `cabal init` looks in the project directory for files that look like
Haskell modules and adds all the modules to the `library:exposed-modules` field.
For modules that do not form part of your package's public interface, you can
move those modules to the `other-modules` field. **Either way, all modules in
the library need to be listed**.

For an executable, `cabal init` does not try to guess which file contains your
program's `Main` module. You will need to fill in the `executable:main-is` field
with the file name of your program's `Main` module. Other modules included in the
executable should be listed in the [`other-modules`](https://cabal.readthedocs.io/en/3.6/cabal-package.html#pkg-field-other-modules) field.

#### 2.1.2 Modules imported from other packages

You have to list all the other packages that your package depends on.

For example, suppose the example `Proglet` module imports the module `Data.Map`.
The `Data.Map` module comes from the `containers` package, so we must list it:

```cabal
library
  exposed-modules:     Proglet
  other-modules:
  build-depends:       containers, base == 4.*
```

We can give a constraint on the version of packages. The most common kinds of
constraints are:

+ `pkgname >= n`
+ `pkgname ^>= n`
+ `pkgname >= n && < m`
+ `pkgname == n.*`

Also, you can factor out shared `build-depends` into a `common` stanza which you
can `import` in your libraries and executable sections. For example

```cabal
common shared-properties
  default-language: Haskell2010
  build-depends:
    base == 4.*
  ghc-options:
    -Wall

library
  import: shared-properties
  exposed-modules:
    Proglet
```

### 2.2 Package concepts

#### 2.2.1 The point of packages

In the Haskell world, packages are not a part of the language itself. Haskell
programs consist of a number of modules, and packages just provide a way to
partition the modules into sets of related functionality.

#### 2.2.2 Kinds of package

There are kinds of package:

+ Cabal
+ GHC
+ system

##### Cabal packages

Cabal packages are really source packages. Cabal packages can be compiled to
produce GHC packages. They can also be translated into operating system packages.

##### GHC packages

GHC only cares about library packages, not executables. Library packages have to
be registered with GHC for them to be available.

The low-level tool `ghc-pkg` is used to register GHC packages and to get information
on what packages are currently registered.

You never need to make GHC packages manually. When you build and install a Cabal
package containing a library then it gets registered with GHC automatically.

##### Operating system packages

On operating systems like Linux, the system has a specific notion of a package
and there are tools for installing and managing packages.

#### 2.2.3 Explicit dependencies and automatic package management

Cabal takes the approach that all packages dependencies are specified explicitly
and specified in a declarative way. The point is to enable automatic package
management.

It is important to track dependencies accurately so that packages can reliably
be moved from one system to another system and still be able to build it there.

## 3. Nix-style Local Builds

Nix-style local builds combine the best of non-sandboxed and sandboxed Cabal:

+ Like sandboxed Cabal previously, we build sets of independent local packages
deterministically and independent of any global state. v2-build will never tell
you that it can't build your package because it would result in a "dangerous
reinstall".
+ Like non-sandboxed Cabal today, builds of external packages are cached
in `~/.cabal/store`, so that a package can be built once, and then reused
anywhere else it is also used.

### 3.1 Quick Start

Suppose that you are in a directory containing a single Cabal package which you
wish to build. You can configure and build it using Nix-style local builds with
this command:

```sh
cabal v2-build
```

To open a GHCi shell with this package, use this command:

```sh
cabal v2-repl
```

To run an executable defined in this package, use this command:

```sh
cabal v2-run <executable name> [executable args]
```

#### 3.1.1 Developing multiple packages

Many Cabal projects involve multiple packages which need to be built together.
To build multiple Cabal packages, you need to first create a `cabal.project` file
which declares where all the local package directories live.

```cabal
packages: Cabal/
          cabal-install/
```

Then to build every component of every package, from the top-level directory,
run the command:

```sh
cabal v2-build all
```

To build a specific package, you can either run `v2-build` from the directory of
the package in question:

```sh
cd cabal-install
cabal v2-build
```

You can also pass the name of the package as an argument to `cabal v2-build`:

```sh
cabal v2-build cabal-install
```

### 3.2 How it works

A **local package** is one that is listed explicitly in the `packages`, `optional-packages`
or `extra-packages` field of a project.

Local packages, as well as the external packages which depend on them, are built
**in place**, meaning that they are built specifically for the project and are
not installed globally.

An **external package** is any package which is not listed in the `packages` field.
The source code for external packages is usually retrieved from Hackage.

When an external package does not depend on an in place package, it can be built
and installed to a **global** store, which can be shared across projects.
The global package store is `~/.cabal/store` (configurable via global `store-dir`
option).
