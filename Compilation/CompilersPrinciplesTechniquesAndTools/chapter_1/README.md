# Chapter 1 Introduction

## 1.1. Language Processors

A *compiler* is a program that can read a program in one language and
translate into an equivalent program in another language.

If the target program is an executable machine-language program, it
can be called by the user to process inputs and produce outputs.

An *interpreter* is another common kind of language processor. Instead of
producing a target program as a translation, an interpreter appears to
directly execute the operations specified on inputs supplied by the user.

## 1.2 The Structure of a Compiler

There are two parts of a compiler: *analysis* and *synthesis*.

The *analysis* part breaks up the source program into constituent pieces and
imposes a grammatical structure on them. It then uses this structure to create
an intermediate representation of the source program. The analysis part also
collects information, about the source program and stores it in a data structure
called a *symbol table*.
