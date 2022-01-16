# DWARF

DWARF is most commonly associated with the ELF object file format, it is
independent of the object file format. It can be used with other object
file format. DWARF does not duplicate information that is contained in
the object file.

## Debugging Information Entry (DIE)

### Tags and Attributes

The basic descriptive entity in DWARF is the Debugging Information Entry (DIE).
A DIE has a *tag*, which specifies what the DIE describes and a list of *attributes*
which fill in details and further describes the entity.

A DIE is contained in or owned by a parent DIE and may have sibling DIEs or children
DIEs. Attributes may contain a variety of values: constants, variables, or references
to another DIE.

Figure 1 shows C's classic `hello.c` program with a simplified graphical representation
of its DWARF description.

![Figure 1. Graphical representation of DWARF data](https://s2.loli.net/2022/01/16/BrnWsTtyd76G21f.png)

The topmost DIE represents the compilation unit. It has two "children", the first is the
DIE describing `main` and the second describing the base type `int` which is the type of
the value returned by `main`.

### Type of DIEs

DIEs can be split into two general types. Those that describe data including data types
and those that describe functions and other executable code.

## Describing Data and Types

Most programming languages have sophisticated descriptions of data. There are a number of
built-in data types, pointers, various data structures, and usually ways of creating new
data types.

Since DWARF is intended to be used with a variety of languages, it abstracts out the
basics and provides a representation that can be used for all supported language. The
primary types, built directly on the hardware, are the *base types*.

### Base Types

DWARF base types provide the lowest level mapping between the simple data types and how
they are implemented on the target machine's hardware.

For example:

```dwarf
DW_TAG_base_type
  DW_AT_name = int
  DW_AT_byte_size = 4
  DW_AT_encoding = signed
```

### Type Composition

A named variable is described by a DIE which has a variety of attributes, one of which is
a reference to a type definition. Below describes an integer variable named `x`.

```dwarf
<1>: DW_TAG_base_type
       DW_AT_name = int
       DW_AT_byte_size = 4
       DW_AT_encoding = signed
<2>: DW_TAG_variable
       DW_AT_name = x
       DW_AT_type = <1>
```

DWARF uses the base types to construct other data type definitions by composition. A new
type is created as a modification of another type.

```dwarf
<1>: DW_TAG_variable
       DW_AT_name = px
       DW_AT_type = <2>
<2>: DW_TAG_pointer_type
       DW_AT_byte_size = 4
       DW_AT_type = <3>
<3>: DW_TAG_base_type
       DW_AT_name = int
       DW_AT_byte_size = 4
       DW_AT_encoding = signed
```

### Array

The index for the array is represented by a *sub-range type* that gives the lower and upper
bounds of each dimension.

### Structures, Classes, Unions, and Interfaces

The DIE for a *class* is the parent of the DIEs which describe each of the class's *members*.
Each *class* has a name and possibly other attributes. If the size of an instance is known at
compile time, then it will have a byte size attribute.

### Variables

Variables are generally pretty simple. They have a name which represents a chunk of memory that
can contain some kind of a value.

For example, the DWARF description of `const char **argv`:

```dwarf
<1>: DW_TAG_variable
       DW_AT_name = argv
       DW_AT_type = <2>
<2>: DW_TAG_pointer_type
       DW_AT_byte_size = 4
       DW_AT_type = <3>
<3>: DW_TAG_pointer_type
       DW_AT_byte_size = 4
       DW_AT_type = <4>
<4>: DW_TAG_const_type
       DW_AT_type = <5>
<5>: DW_TAG_base_type
       DW_AT_name = char
       DW_AT_byte_size = 1
       DW_AT_encoding = unsigned
```

## Location Expressions

DWARF provides a very general scheme to describe how to locate the data represented by a
variable. A DWARF location expression contains a sequence of operations which tell a
debugger how to locate the data.

```c
int a;
void foo() {
  register int b;
  int c;
}
```

For example, variable `a` has a fixed location in memory, variable `b` is in register 0,
and variable `c` is at offset -12 within the *current function's stack frame*. Although
`a` was declared first, the DIE to describe it is generated later, after all functions.

```dwarf
<1>: DW_TAG_subprogram
       DW_AT_name = foo
<2>: DW_TAG_variable
       DW_AT_name = b
       DW_AT_type = <4>
       DW_AT_location = (DW_OP_reg0)
<3>: DW_TAG_variable
       DW_AT_name = c
       DW_AT_type = <4>
       DW_AT_location = (DW_OP_fbreg: -12)
<4>: DW_TAG_base_type
       DW_AT_name = int
       DW_AT_byte_size = 4
       DW_AT_encoding = signed
<5>: DW_TAG_variable
       DW_AT_name = a
       DW_AT_type = <4>
       DW_AT_external = 1
       DW_AT_location = (DW_OP_addr: 0)
```

## Describing Executable Code

### Functions and Subprograms

DWARF treats functions that return values and subroutines that do not as variations of
the same thing. DWARF describes both with a subprogram DIE. This IDE has a name, a source
location triplet, and an attribute which indicates whether the subprogram is *external*.

A subprogram DIE has attributes that give the low and high memory addresses that the
subprogram occupies, if it is contiguous.

The value that a function returns is given by the *type* attribute. Subroutines that do
not return values do not have this attribute. The *return address* attribute is a location
expression that specifies where the address of the caller is stored. The *frame base*
attribute is a location expression that computes the address of the stack frame for the
function.

### Compilation Unit

DWARF calls each separately compiled source file a compilation unit.

The DWARF data for each compilation unit starts with a Compilation Unit DIE. This DIE
contains general information about the compilation, including the directory and name of
the source file, the programming language used, a string which identifies the producer
of the DWARF data, and offsets into the DWARF data sections to help locate the line number
and macro information.

## Other DWARF Data

### Line Number Table

The DWARF line table consists the mapping between memory addresses that contain the executable
code of a program and the source lines that corresponding to these addresses. In the simplest
form, this can be looked at as a matrix with one column containing the memory addresses and
another column containing the source triplet (file, line, and column) for that address.

It may be useful to identify the end of the code which represents the prolog of a function or
the beginning of the epilog, so that the debugger can stop after all of the arguments to a
function have been loaded or before the function returns.

### Macro Information

DWARF includes the description of the macros defined in the program. This is quite rudimentary
information, but can be used by a debugger to display the values for a macro or possibly
translate the marco into the corresponding source language.

### Call Frame Information

The DWARF Call Frame Information (CFI) provides the debugger with enough information about how
a function is called so that it can locate each of the arguments to the function, locate the
current call frame, and locate the call frame for the calling function. This information is used
by the debugger to "unwind the stack".
