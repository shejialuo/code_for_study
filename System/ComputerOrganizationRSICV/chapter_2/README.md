# Chapter 2. Instructions: Language of the Computer

## 2.1 Introduction

The words of a computer's language are called *instructions*, and its
vocabulary is called an *instruction set*.

## 2.2 Operations of the Computer Hardware

Every computer must be able to perform arithmetic. The RISC-V assembly
language notation `add a, b, c` instructs a computer to add the two variables
`b` and `c` to put their sum in `a`.

The natural number of operands of an operation like addition is three:
the numbers being added together and a place to put the sum. Requiring
every instruction to have exactly three operands, no more or no less,
conforms to the philosophy of keeping the hardware simple.

*Design Principle 1*: Simplicity favors regularity.

## 2.3 Operands of the Computer Hardware

The operands of arithmetic instructions are restricted; they must be from
a limited number of special locations built directly in hardware called *registers*.
The size of a register that RISC-V architecture is 32 bits named `word`, for 64 bits
named `doubleword`

There are only 32 registers in RISC-V.

*Design Principle 2*: Smaller is faster.

RISC-V must include instructions that transfer data between memory
and registers. Such instructions are called *data transfer instructions*.

```assembly
lw x9, 32(x22)   // load word
add x9, x21, x9
sw x9, 48(x22)   // save word

addi x22, x22, 4 // i means immediate
```

## 2.4 Signed and Unsigned Numbers

Computer programs calculate both positive and negative numbers, so we
need a representation that distinguishes the positive form the negative.
The most obvious solution is to add a separate sign, which conveniently can be
represented in a single bit; the name of this representation is
*sign and magnitude*.

However, there are several shortcomings:

+ Where to put the sign bit?
+ Adders for sign and magnitude may need an extra step to set the sign.
+ A separate sign bit means that there is both a positive and negative zero.

The final solution is to pick the representation that makes the hardware simple:
leading 0s mean positive, and leading 1s mean negative. This convention
for representing signed binary numbers is called *two's complement*
representation.

Two's complement representation has the advantage that all negative
numbers have a 1 in the most significant bit. Thus, hardware needs to test
only this bit to see if a number is positive or negative.
This bit is often called the *sign bit*.

$$
x_{31} \times -2^{31} + x_{30} \times 2^{30} + \cdots + x_{0} \times 2^{0}
$$

Let's examine two useful shortcuts when working with two's complement numbers.
The first shortcut is a quick way to negate a two's complement binary
number. Simply invert every 0 to 1 and every 1 to 0, then add one to the result.
This shortcut is based on the observation that the sum of a number and
its inverted representation must be $111 \dots 111_{two}$, which
represents -1. Since $x + \bar{x} = -1$, so we can get $\bar{x} + 1 = -x$.

Signed versus unsigned applies to loads as well as to arithmetic. The *function* of
a signed load is to copy the sign repeatedly to fill the rest
of the register: called *sign extension*. Unsigned loads simply fill
with 0s to the left of the data.

When loading a 32-bit word into a 32-bit register, the point it moot; signed
and unsigned loads are identical. RISC-V offers two flavors of byte loads:
*load byte unsigned*(`lbu`) treats the byte as an unsigned number
and thus zero-extends to fill the leftmost bits of the register,
while *load byte*(`lb`) works with signed integers.

## 2.5 Representing Instructions in the Computer

Instructions are kept in the computer as a series of high and low
electronic signals and may be represented as numbers.

The layout of the instruction is called the *instruction format*. RISC-V
instruction takes exactly 32 bits in keeping with our design principle:
simplicity favors regularity.

To distinguish it from assembly language, we call the numeric version of
instructions *machine language* and a sequence of such instructions *machine code*.

### 2.5.1 RISC-V Fields

RISC-V fields are given names to make them easier to discuss.

![RISC-V R Type](https://s2.loli.net/2022/04/19/fQdTFXK4PUwnj7b.png)

Here is the meaning of each name of the fields in RISC-V instructions:

+ opcode: Basic operation of the instruction.
+ rd: The register destination operand. It gets the result of the operation.
+ funct3: An additional opcode field.
+ rs1: The first register source operand.
+ rs2: The second register source operand.
+ funct7: An additional opcode field

A problem occurs when an instruction needs longer fields than those above.
For example, the load register instruction must specify two registers
and a constant. If the address were to use one of the 5-bit fields in the
format above, the largest constant within the load register would be limited
to only 31.

Hence, we have a conflict between the desire to keep all instructions the same
length and the desire
to have a single instruction format. This conflict leads us to the
final design principle:

*Design Principle 3*: Good design demands good compromises.

This compromise chosen by the RISC-V designers is to keep all the instructions
the same length, thereby requiring distinct instruction formats
for different kinds of instructions. For example, the format above is called *R-type*.
A second type of instruction format is *I-type* and is used by arithmetic
operands with one constant operand, including `addi`. The idea is simple,
just combine funct7 and rs2 to become immediate.

![RISC-V I Type](https://s2.loli.net/2022/04/19/IW9ekvKBoPfa1Tw.png)

We also need a format for the store word instruction, `sw`, which needs two
source registers and an immediate for the address offset. The fields of
S-type format are

![RISC-V S Type](https://s2.loli.net/2022/04/19/GpVDAbMXlu7tBYK.png)

The formats are distinguished by the values in the opcode field:
each format is assigned a distinct set of opcode values in the first
field so that the hardware knows how to treat the rest of the instruction.

## 2.6 Logical Operations

+ Shift left: `sll`, `slli`
+ Shift right: `srl`, `srli`
+ Shift right arithmetic: `sra`, `srai`
+ Bit-by-bit AND: `and`, `andi`
+ Bit-by-bit OR: `or`, `ori`
+ Bit-by-bit XOR: `xor`, `xori`
+ Bit-by-bit NOT: `xori`

## 2.7 Instructions for Making Decisions

RISC-V assembly language includes two decision-making instructions, similar
to an `if` statement with a `go-to`. The first instruction is

```assembly
beq rs1, rs2, L1
```

This instruction means go to the statement labeled `L1` if the
value in register `rs1` equals the value in register `rs2`. The mnemonic
`beq` stands for *branch if equal*. The second instruction is

```assembly
bne rs1, rs2, L1
```

Thus, we can make a loop:

```assembly
Loop:
slli x10, x22, 2
add x10, x10, x25
lw x9, 0(x10)
bne x9, x24, Exit
addi x22, x22, 1
beq x0, x0 Loop
Exit
```

+ Branch if less than: `blt`
+ Branch if greater than or equal: `bge`
+ Branch if less than, unsigned: `bltu`
+ Branch if greater than or equal: `bgeu`

Yet another alternative, used by ARM's instruction sets, is to keep
extra bits that record what occurred during an instruction. These
additional bits, called *condition codes* or *flags*.

## 2.8 Supporting Procedures in Computer Hardware

RISC-V software follows the following convention for procedure for
procedure calling in allocating its 32 registers:

+ `x10-x17`: eight parameter registers in which to pass parameters or
return values.
+ `x1`: one return address register to return to the point of origin.

In addition to allocating these registers, RISC-V assembly language
includes an instruction just for the procedures: it branches to an address and
simultaneously saves the address of the following instruction to the
destination register `rd`, named *jump-and-link instruction* (`jal`).

```assembly
// jump to ProcedureAddress and write return address to x1
jal x1, ProcedureAddress
```

To support the return from a procedure, computers like RISC-V use an
indirect jump, like the jump-and-link instruction (`jalr`).

The jump-and-link register instruction branches to the address stored
in register `x1`. Thus, the caller puts the parameter values in `x10-x17` and
uses `jal x1, X` to branch to procedure `X`. The callee then performs
the calculations, places the results in the same parameter registers,
and return control to the caller using `jalr x0, 0(x1)`.

### 2.8.1 Using More Registers

In RISC-V, the *stack pointer* is register `x2`, also known by the name
`sp`.

By historical precedent, stacks "grow" from higher addresses to lower
addresses. This convention means that you push values onto the stack
by subtracting from the stack pointer. Adding to the stack pointer shrinks the stack,
thereby popping values off the stack.

```c
int leaf_example(int g, int h, int i, int j) {
  int f;
  f = (g + h) - (i + j);
  return f;
}
```

The parameter variables `g`, `h`, `i` and `j` correspond to the
argument registers `x10`, `x11`, `x12` and `x13`, respectively, and
`f` corresponds to `x20`.

```assembly
leaf_example:
  addi sp, sp, -12
  sw x5, 8(sp)
  sw x6, 4(sp)
  sw x20, 0(sp)

  add x5, x10, x11 // g + h
  add x6, x12, x13 // i + j
  sub x20, x5, x6  // f = (g + h) - (i + j)

  addi x10, x20, 0

  lw x20, 0(sp)  // restore for caller
  lw x6, 4(sp)   // restore for caller
  lw x5, 8(sp)   // restore for caller
  addi sp, sp, 12

  jalr x0, 0(x1) // return
```

RISC-V software separates 19 of the registers into two groups:

+ `x5-x7` and `x28-x31`: temporary registers that are *not* preserved
by the callee on a procedure call.
+ `x8-x9` and `x18-x27`: saved registers that must be preserved on a
procedure call.

### 2.8.2 Nested Procedures

```c
int fact(int n) {
  if(n < 1) return 1;
  else return (n * fact(n - 1));
}
```

```assembly
fact:
  adi sp, sp, -8
  sw x1, 4(sp)  // save the return address
  sw x10, 0(sp) // save the argument n

  addi x5, x10, -1 // x5 = n -1
  bge x5, x0, L1   // if (n - 1) >= 0, go to L1

  addi x10, x0, 1  // return 1
  addi sp, sp, 8
  jalr x0, 0(x1)  // return to caller
L1:
  addi x10, x10, -1 // n - 1
  jal x1, fact  //

  addi x6, x10, 0 // x6 = fact(n - 1)
  lw x10, 0(sp)
  lw x1, 4(sp)
  addi sp, sp, 8
  mul x10, x10, x6
  jalr x0, 0(x1)
```

### 2.8.3 Allocating Space for New Data on the Stack

Some RISC-V compilers use a *frame pointer* `fp`, or register `x8` to point
to the first word of the frame of a procedure. A stack pointer might change
during the procedure, and so references to a local variable in memory
might have different offsets depending on where they are in the procedure,
making the procedure harder to understand. Alternatively, a frame pointer offers a stable base register within a procedure
for memory-references.

## 2.9 Communicating with People

Most computers today offer 8-bit bytes to represent characters, with the ASCII.

## 2.10 RISC-V Addressing for Wide Immediates and Addresses

Although keeping all RISC-V instructions 32 bits long simplifies the hardware,
there are times where it would be convenient to have 32-bit or larger
constants or address.

### 2.10.1 Wide Immediate Operands

The RISC-V instruction set includes the instruction *Loader upper immediate*(`lui`) to
load a 20-bit constant into bits 12 through 31 of a register. The rightmost 12bits
are filled with zeros.

```assembly
lui x19, 976
addi x19, x19, 1280
```

Either the compiler or the assembler must break large constants into
pieces and then reassemble them into a register. As you might expect,
the immediate field's size restriction may be a problem for memory addresses in
loads and stores
as well as for constants in immediate instructions.

### 2.10.2 Addressing in Branches

The RISC-V branch instructions use an RISC-V instruction format
with a 12-bit immediate. The SB-type format consists of a 7-bit opcode, a 3-bit
function code, two 5-bit register operands, and a 12-bit address
immediate. The address uses an unusual encoding, which simplifies
datapath design but complicates assembly. The instruction could
be assembled into the S format.

The unconditional jump-and-link instruction uses an instruction with
a 12-bit immediate. This instruction consists of a 7-bit opcode, a
5-bit destination register operand, and 20-bit address immediate.

Like the SB-type format, the UJ-type format's address operand uses
an unusual immediate encoding. So it could be assembled into the U format.

RISC-V uses PC-relative addressing for both conditional branches and
unconditional jumps.
