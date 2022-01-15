# Chapter 2 Detecting the Environment

We may find it necessary to configure and/or build code slightly
differently depending on the platform.

## 1. Discovering the operating system

CMake correctly defines `CMAKE_SYSTEM_NAME` for the target OS and
therefore there is typically no need to use custom commands, tools,
or scripts to query this information. On systems that have the `uname`
command, this variable is set to the output of `uname -s`.

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Configuring on/for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "Configuring on/for macOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "Configuring on/for Windows")
endif()
```

## 2. Dealing with platform-dependent source code

Ideally, we should avoid platform-dependent source code, but sometimes
we have no choice.

```c++
std::string say_hello(){
#ifdef IS_WINDOWS
  return std::string("Hello from Windows!");
#ifdef IS_LINUX
  return std::string("Hello from Linux!");
}
```

We could use `target_compile_definitions(<EXECUTABLE_NAME> <OPTIONS> <DEFINITION>)`.

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(hello-world PUBLIC "IS_LINUX")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_compile_definitions(hello-world PUBLIC "IS_Windows")
endif()
```

## 3. Dealing with compiler-dependent source code

We could use `CMAKE_CXX_COMPILER_ID MATCHES <NAME>`.

## 4. Discovering the host processor architecture

We could use `CMAKE_HOST_SYSTEM_PROCESSOR MATCHES <NAME>`.
