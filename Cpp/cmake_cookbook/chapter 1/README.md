# Chapter 1 From a Simple Executable to Libraries

## 1. Compiling a single source file into an executable

We wish to compile the following source code into a single executable:

```c++
#include <cstdlib>
#include <iostream>
#include <string>

std::string say_hello() {return std::string("Hello, CMake world!");}

int main() {
  std::cout << say_hello() << std::endl;
  return EXIT_SUCCESS;
}
```

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(recipe-01 LANGUAGES CXX)
add_executable(hello-world hello-world.cpp)
```

For convenience, we could use `cmake -H. -Bbuild`.

CMake is a build system *generator*. You describe what type of operations the build system, such as Unix Makefiles,
Ninja, Visual Studio, and so on. In turn, CMake *generates* the corresponding instructions for the chosen
build system.

Note the `cmake --build . --target <target-name>`:

+ `all` is the default target and will build all other targets in the project.
+ `clean` is the target to choose if one wants to remove all generated files.
+ `depend` will invoke CMake to generate the dependencies.
+ `rebuild_cache` will once again invoke `CMake` to rebuild the `CMakeCache.txt`.
+ `edit_cache` will let you edit cache entries directly.

For more complex projects, with a test stage and installation rules, CMake will generate
additional convenience targets:

+ `test` will run the test suite with the help of CTest.
+ `install` will execute the installation rules for the project.
+ `package` will invoke CPack to generate a redistributable package for the project.

## 2. Switching generators

CMake will generate the corresponding instructions for the chosen build system.

If we want to switch generator, we will have to pass the generator explicitly with the `-G` CLI switch.

## 3. Building and linking static and shared libraries

We could use `add_library(<LIB_NAME> <OPTION> <SOURCE_FILE>)` to generate the necessary build tool
instructions for compiling the specified sources into a library. The actual name of the generated library
will be formed by CMake by adding the prefix `lib` in front and the appropriate extension as a suffix.

We could use `target_link_libraries(<EXECUTABLE_NAME> <LIB_NAME>)` to link the library into the executable.

CMake accepts other values as valid for the second `OPTION` argument to
`add_library`:

+ `STATIC`
+ `SHARED`
+ `OBJECT` can bse used to compile the sources in the list given to `add_library` to object files, but then
  neither archiving them into a static library nor linking them into a shared object.
  **The use of object libraries is particularly useful if one needs to create both static and shared libraries in one go**.
+ `MODULE` libraries are not linked to any other target within the project, but may be loaded
  dynamically later on.

CMake is also to generate special types of libraries. These produce no output in the build system
but are extremely helpful in organizing dependencies and build requirements between targets:

+ `IMPORTED`, this type of library represents a library located *outside* the project, immutable.
+ `INTERFACE`, like `IMPORTED`, but is mutable and has no location.
+ `ALIAS`

## 4. Controlling compilation with conditionals

CMake offers its own language. We will explore the use of the conditional construct `if-elseif-else-endif`.

```cmake
set(USE_LIBRARY OFF)
message(STATUS "Compile sources into a library? ${USE_LIBRARY}")

if(USE_LIBRARY)
  # do something
else()
  # do something
endif()
```

## 5. Presenting options to the user

We could use `option` to present options to the user:

```cmake
option(<option_variable> "help string" [initial value])
```

For example:

```cmake
option(USE_LIBRARY "Compile sources into a library" OFF)
```

```sh
cmake -D USE_LIBRARY=ON ..
```

Sometimes there is the need to introduce options that are dependent on the value of other options.

```cmake
include(CMakeDependentOption)

# second option depends on the value of the first
cmake_dependent_option(
  MAKE_STATIC_LIBRARY "Compile sources into a static library" OFF
  "USE_LIBRARY" ON
 )

# third option depends on the value of the first
cmake_dependent_option(
  MAKE_SHARED_LIBRARY "Compile sources into a shared library" ON
  "USE_LIBRARY" ON
)
```

## 6. Specifying the compiler

CMake stores compilers for each language in the `CMAKE_<LANG>_COMPILER` variable, where `<LANG>` is any
of the supported languages. The user can set this variable in one of two ways:

1. By using the `-D` option in the CLI, for example:

   ```sh
   cmake -D CMAKE_CXX_COMPILER=clang++ ..
   ```

2. By exporting the environment variables.

## 7. Switching the build type

The variable governing the build type to be used when generating the build system is `CMAKE_BUILD_TYPE`. This
variable is empty by default, and the values recognized by CMake are:

1. `Debug` for building your library or executable without optimization and with debug symbols.
2. `Release` for building your library or executable with optimization and without debug symbols.
3. `RelWithDebInfo` for building your library or executable with less aggressive optimizations and with debug symbols.
4. `MinSizeRel` for building your library or executable with optimizations that do not increase object code size.

## 8. Controlling compiler flags

CMake offers a lot of flexibility for adjusting or extending compiler flags and you can choose
between two main approaches:

+ CMake treats compile options as properties of targets.
+ You can directly modify the `CMAKE_<LANG>_FLAGS_<CONFIG>` variables by using the `-D` CLI switch.

```cmake
target_compile_options(<LIBRARY_NAME> <OPTION> <FLAGS>)
```

Compiler options can be added with three levels of visibility: `INTERFACE`, `PUBLIC`, and `PRIVATE`.

The visibility levels have the following meaning:

+ With the `PRIVATE` attribute, compile options will only be applied to the given target and not to other
  targets consuming it.
+ With the `INTERFACE` attribute, compile options on a given target will only be applied to targets consuming it.
+ With the `PUBLIC` attribute, compile options will be applied to the given target and all other targets consuming it.

Most of the time, flags are compiler-specific. Our current example will only work with GCC and Clang;
compilers from other vendors will not understand many.

```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  list(APPEND CMAKE_CXX_FLAGS "-fno-rtti" "-fno-exceptions")
  list(APPEND CMAKE_CXX_FLAGS_DEBUG "-Wsuggest-final-types")
  list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3" "-Wno-unused")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  #
  #
  #
endif()
```
