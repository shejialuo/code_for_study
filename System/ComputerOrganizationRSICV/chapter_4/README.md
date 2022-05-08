# Chapter 4 Introduction

## 4.1 Introduction

### 4.1.1 A Basic RISC-V Implementation

We will be examining an implementation that includes a subset of the
core RISC-V instruction set:

+ The memory-reference instructions `lw` and `sw`.
+ The arithmetic-logical instructions `add`, `sub`, `and`, and `or`.
+ The conditional branch instruction `beq`.

This subset illustrates the key principles used in creating a datapath
and designing the control.

### 4.1.2 An Overview of the Implementation

For every instruction, the first two steps are identical:

+ Send the *program counter* to the memory that contains the code
and fetch the instruction from that memory.
+ Read one or two registers, using fields of the instruction to
select the registers to read.

After these two steps, the actions required to complete the instruction
depend on the instruction class. Fortunately, for each of the three instruction classes,
the actions are largely the same.

For example, all instruction classes use the arithmetic-logical unit
(ALU) after reading the registers. The memory-reference
instructions for the operation execution, and conditional branches
for the equality test. After using the ALU, the actions required to
complete various instruction classes differ. A memory-reference
instruction will need to access the memory either to read data for a load
or write data for a store. An arithmetic-logical or load instruction
must write the data from the ALU or memory back into a register. Lastly,
for a conditional branch instruction, we may need to change the next
instruction address based on the comparison; otherwise, the
PC should be incremented by four to get the address of the subsequent instruction.

Below shows the high-level view of RISC-V implementation, focusing
on the various functional units and their interconnection.
Although this figure shows most of the flow of data through the processor,
it omits two important aspects of instruction execution.

![An abstract view of the implementation of the RISC-V](https://s2.loli.net/2022/05/05/56gYhJIeSjR7NAM.png)

First, in several places, above figure shows data going to a particular
unit as coming from two different sources. For example, the
value written into the PC can some from one of two adders, the
data written into the register file can come from either the ALU
or the data memory, and the second input to the ALU can come from
a register or the immediate field of the instruction.

Now we shows the datapath with the three required multiplexors added, as
well as control lines for the major functional units. A *control unit*,
which has the instruction as an input, is used to determine how to set
the control lines for the functional units and two of the multiplexors.

![The basic implementation of the RISC-V](https://s2.loli.net/2022/05/05/kQ3UmlAK92dVwLt.png)

## 4.2 Building a Datapath

A reasonable way to start a datapath design is to examine the major
components required to execute each class of RISC-V instructions.

To execute any instruction, we must start by fetching the instruction from
memory. To prepare for executing the next instruction, we must also increment
the program counter so that it points at the next instruction, 4 bytes later.

![A portion of the datapath used for fetching instructions and incrementing the program counter](https://s2.loli.net/2022/05/05/Q2nsLUJqYMrOxN8.png)

Now, let's consider the R-format instructions. They all read two registers,
perform an ALU operation on the contents of the registers, and
write the result to a register.

The processor's 32 general-purpose registers are stored in a structure
called a *register file*. A register file is a collection of registers
in which any register can be read or written by specifying the number
of the register in the file. The register file contains the register
state of the computer. In addition, we will need an ALU to operate on the
values read from the register.

For each data word to be read from the registers, we need an input to the
register file that specifies the *register number* to be read and an output
from the register file that will carry the value that has been read from
the registers. To write a data word, we will need two inputs: one to
specify the register number to be written and one to supply the *data*
to be written into the register. The register file always outputs the
contents of whatever register numbers are on the Read register inputs.
Writes, however, are controlled by the write control signal, which must
be asserted for a write to occur at the clock edge. Below shows the result;
we need a total of three inputs (two for register numbers and one for data)
and two outputs (both for data). The register number inputs are
5 bits wide to specify one of 32 registers, whereas the data input
and two data output buses are each 32 bits wide

Also the ALU takes two 32-bit inputs and produces a 32-bit result,
as well as a 1-bit signal if the result is 0.

![The two elements needed to implement R-format ALU operations](https://s2.loli.net/2022/05/05/eE5sqB1nfMSz6bU.png)

Next, consider the RISC-V load register and store register instructions,
which have the general form `lw x1, offset(x2)` or `sw x1, offset(x2)`.
We will need both the register file and the ALU. Furthermore, we will
need a unit to *sign-extend* the 12-bit offset field in the instruction
to a 32-bit signed value, and a data memory unit to read from or write to.
The data memory must be written on store instructions; hence, data
memory has read and write control signals, an address input, and an
input for the data to be written into memory.

![The two units needed to implement loads and stores](https://s2.loli.net/2022/05/05/RQ4xWsGX2V8Z3k6.png)

The `beq` instruction has three operands, two registers that are
compared for equality, and a 12-bit offset used to compute the
*brand target address* relative to the branch instruction address.
To implement this instruction address, we must compute the branch
target address by adding the sign-extended offset field of the instruction
to the PC. There are two details in the definition of branch instructions
to which we must pay attention:

+ The instruction set architecture specifies that the base for the
branch address calculation is the address of the branch instruction.
+ The architecture also states that the offset field is shifted left 1 bit.

As well as computing the branch target address, we must also determine
whether the next instruction is the instruction that follows sequentially
or the instruction at the branch target address. When the condition is true,
the branch target address becomes the new PC, and we say that the
*branch* is *taken*. If the operand is not zero, the incremented PC
should replace the current PC; in this case, we say that the
*branch* is *not taken*.

Thus, the branch datapath must do two operations: compute the branch
target address and test the register contents. Below shows the structure
of the datapath segment that handles branches

![The portion of a datapath for a branch](https://s2.loli.net/2022/05/05/LWgZBM7NVrkP5uc.png)

### 4.2.1 Creating a Single Datapath

Now that we have examined the datapath components needed for the individual
instruction classes, we can combine them into a single datapath and
add the control to complete the implementation. The simplest datapath will attempt
to execute all instructions in one clock cycle. Thus, that no datapath
resource can be used more than once per instruction, so any element
needed more than once per instruction, so any element needed more than
once must be duplicated. We therefore need a memory for instructions separate
from one for data.

To share a datapath element between two different instruction classes,
we may need to allow multiple connections to the input of an element,
using a multiplexor and control signal to select among the multiple inputs.

![The simple datapath for the core RISC-V architecture](https://s2.loli.net/2022/05/05/jMkEKWBxe2wI1oG.png)

## 4.3 A simple Implementation Scheme

In this section, we look at what might be though of as a simple implementation
of our RISC-V subset.

### 4.3.1 The ALU Control

The RISC-V ALU defines the four following combinations of our
control inputs

| ALU control lines | Function |
|:-----------------:|:--------:|
|        0000       |    AND   |
|        0001       |    OR    |
|        0010       |    add   |
|        0110       | subtract |

Depending on the instruction class, the ALU will need to perform
one of these four functions. For load and store instructions, we use
the ALU to compute the memory address by addition. For the R-type instructions,
the ALU needs to perform one of the four actions, depending on the
value of the 7-bit `funct7` field and 3-bit `funct3` field in the instruction.
For the conditional branch if equal instruction, the ALU subtracts two
operands and tests to see if the result is 0.

We can generate the 4-bit ALU control input using a small control unit
that has as inputs the `funct7` and `funct3` fields of the instruction and
a 2-bit control field, which we call `ALUOp`. `ALUOp` indicates whether
the operation to be performed should be add, subtract and test, or be
determined by the operation encoded in the `funct7` and `funct3` fields.

![How the ALU control bits are set](https://s2.loli.net/2022/05/08/KXHB4pzPjt1GyOL.png)

### 4.3.2 Designing the Main Control Unit

There are several major observations about instruction format
that we will rely on:

![The four instruction classes use four different instruction format](https://s2.loli.net/2022/05/08/CcSJgHxUOPpW6Iy.png)

+ The `opcode` field is always in bits 6:0. Depending on the `opcode`,
the `funct3` field(bits 14:12) and `func7` field(bits 31:25) serve
as an extended `opcode` field.
+ The first register operand is always in bit positions 19:15(`rs1`)
for R-type instructions. This field also specifies the base register for
load and store instructions.
+ The second register operand is always in bit positions 24:20(`rs2`)
for R-type instructions and branch instructions. This field also
specifies the register operand that gets copied to memory for
store instructions.
+ Another operand can also be a 12-bit offset for branch or load-store instructions.
+ The destination register is always in bit positions 11:7(`rd`) for R-type
instructions and load instructions.

Below shows six single-bit control lines plus the 2-bit `ALUOp` control signal.

![The datapath with all necessary multiplexors and all control lines identified](https://s2.loli.net/2022/05/08/GaMWi7REupSJqUD.png)

![The effect of each of the six control signals](https://s2.loli.net/2022/05/08/Lnr2HiNOuvPbkls.png)
