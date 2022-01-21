# Chapter 3 Detecting External Libraries and Programs

In this chapter, we will discuss the necessary tools and discover the find family of CMake commands:

+ `find_file` to find a full path to a named file
+ `find_library` to find a library
+ `find_package` to find and load settings from an external project
+ `find_path` to find a directory containing the named file
+ `find_program` to find a program

## 1. Detecting the Python interpreter

```cmake
find_package(PythonInterp REQUIRED)
```

`find_package` is a wrapper command for CMake modules written for discovering and setting up packages.
These modules contain CMake commands to identify packages in standard locations on the system.
The files for the CMake modules are called `Find<name>.cmake` and the commands they contain will be
run internally when a call to `find_package(<name>)` is issued.

In addition to actually discovering the requested package on your system, find modules also set up a handful
of useful variables.

Sometimes, packages are not installed in standard locations and CMake might fail to locate them correctly.
It is possible to tell CMake to look into certain specific locations to find certain software using the CLI
switch `-D` to pass the appropriate option.

## 2. Detecting the Python library

Sometimes, we want to embed the Python interpreter into a C or C++ program, which requires:

+ A working version of the Python interpreter
+ The availability of the Python header file `Python.h`
+ The python runtime library `libpython`

```cmake
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
target_include_directories(hello-embedded-python
  PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hello-embedded-python
  PRIVATE ${PYTHON_LIBRARIES})
```

Using the `EXACT` keyword, we have constrained CMake to detect a particular and in this case matching,
version of the Python include files and libraries.

## 3. Detecting Python modules and packages

NumPy has become very popular in the scientific community for problems involving matrix algebra.
In projects that depend on Python modules or packages, it is important to make sure
that the dependency on these Python modules is satisfied.

We could use `execute_process` to find the `NumPy` location. The `execute_process` command will
execute one or more commands as child processes to the currently issued CMake command.
The return value for the last child process will be saved into the variable passed as
an argument to `RESULT_VARIABLE`, while the contents of the standard output and standard error
pipes will be saved into the variables passed as arguments to `OUTPUT_VARIABLE`.

