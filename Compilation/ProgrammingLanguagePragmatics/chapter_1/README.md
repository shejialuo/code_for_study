# Chapter 1 Introduction

## 1.1 The Art of Language Design

Simple. Omit detail here.

## 1.2 The Programming Language Spectrum

The many existing languages can be classified into families based on
their model of computation. Below figure shows a common set of families.
The top-level division distinguishes between the *declarative* languages,
in which the focus is on *what* the computer is to do, and the *imperative* languages,
in which the focus is on *how* the computer should do it.

![Classification of programming languages](https://s2.loli.net/2022/01/19/LMTSaNg3IbemWVw.png)

Within the declarative and imperative families, there are several
important subfamilies:

+ *Functional* languages employ a computational model based on
the recursive definition of functions.
+ *Dataflow* languages model computation as the flow of the information (tokens)
among primitive functional *nodes*.
+ *Logic* or *constraint-based* languages take their inspiration from predicate logic.
They model computation as an attempt to find values that satisfy certain
specified relationships.
+ The *von Neumann* languages.
+ The *object-oriented* languages.
+ *Scripting* languages.

## 1.3 Why Studying Programming Languages?

Whatever language you learn, understanding the decisions that went
into its design and implementation will help you use it better.

+ *Understand obscure features*.
+ *Choose among alternative ways to express things*.
+ *Make good use of debuggers, assemblers, linkers and related tools*.
+ *Simulate useful features in languages that lack them*.
+ *Make better use of language technology wherever it appears*.

## 1.4 Compilation and Interpretation

At the highest level of abstraction, the compilation and execution
of a program in a high-level language look something like this:

![Pure compilation](https://s2.loli.net/2022/01/19/YTyXRC9dJKge7zL.png)

The compiler *translates* the high-level source program into an equivalent
target program, and then goes away.

The compiler is the locus of control during compilation; the target
program is the locus of control during its own execution.

An alternative style of implementation of high-level languages is known
as *interpretation*:

![Pure interpretation](https://s2.loli.net/2022/01/19/8Wi412ReIkwqVFu.png)

Unlike a compiler, an interpreter stays around for the execution of the
application. In fact, the interpreter is the locus of control during that
execution. In effect, the interpreter implements a virtual machine whose
"machine language" is the high-level programming language. The interpreter reads
statements in that language more or less one at a time, executing them
as it goes along.

In general, interpretation leads to greater flexibility and better
diagnostics than does compilation. Compilation generally leads to
better performance.

While the conceptual difference between compilation and interpretation is
clear, most language implementation include a mixture of both. They typically
look like this:

![Mixing compilation and interpretation](https://s2.loli.net/2022/01/19/IdzFZAa1YmnW6TV.png)

In practice one sees a broad spectrum of implementation strategies:

+ Most interpreted languages employ an initial translator (a *preprocessor*)
that removes comments and white space, and group characters together into *tokens*
such as keywords, identifiers, numbers, and symbols. The translator may also expand
abbreviations in the style of a macro assembler. Finally, it may identify
higher-level syntactic structures, such as loops and subroutines. The goal is to
**produce an intermediate form that mirrors the structure of the source**,
but can be interpreted more efficiently.

+ Many compilers generate assembly language instead of machine language.
This convention facilitates debugging.

+ Many compilers are *self-hosting*: they are written in the language they compile.
This raises an obvious question: how does one compile the compiler in the fist place?
The answer is to use a technique known as *bootstrapping*.

+ One will sometimes find compilers for languages that permit a lot of late
binding, and are traditionally interpreted.

+ In some cases a programming system may deliberately delay compilation until the
last possible moment. The Java language definition defines a machine-independent intermediate
form known as Java *bytecode*. Bytecode is the standard format for distribution
of Java programs; it allows programs to be transferred easily over the
Internet, and then run on any platform. The first Java implementations were based on
byte-code interpreters, but modern implementations obtain significantly better
performance with a *just-in-time* compiler that translates bytecode into
machine language immediately before each execution of the program.

## 1.5 Programming Environments

Simple. Omit.

## 1.6 An overview of Compilation

In a typical compiler, compilation proceeds through a series of well-defined *phases*.

![Phases of compilation](https://s2.loli.net/2022/01/19/bcV7gnOCT8LZ1MY.png)

The first few phases serve to figure out the meaning of the source program. They are sometimes
called the *front end* of the compiler. The last few phases serve to construct
an equivalent target program. They are sometimes called the *back end* of the compiler.

An interpreter shares the compiler's front-end structure, but "executes"
(interprets) the intermediate form directly, rather than translating it into
machine language. The execution typically takes the form of a set of mutually
recursive subroutines that traverse the syntax tree, "executing" its node in program order.

![Phases of interpretation](https://s2.loli.net/2022/01/19/HYSbu1Le2mfviOE.png)
