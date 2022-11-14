# Native Client: A Sandbox for Portable, Untrusted x86 Native Code

## Introduction

The background is that sometimes web applications want to run some native code to
improve the performance. To ensure the safety, this paper comes.

## System Architecture

It consists of two components: A user interface implemented in JavaScript and
executing in the web browser, and an image processing library, implemented as a
NaCl module.

When the user navigates to the web site that hosts the photo application, the browser
loads and executes the application JavaScript components. The JavaScript in turn invokes
the NaCl browser plugin to load the image processing library into a NaCl container.

Each component runs in its own private address space. Inter-component communication
is based on Native Client's reliable diagram service.

## Conclusion

Sandbox technology
