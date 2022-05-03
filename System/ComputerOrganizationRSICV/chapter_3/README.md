# Chapter 3 Arithmetic for Computers

## 3.1 Introduction

+ What about fractions and other real numbers.
+ What happens if an operation creates a number bigger than can be represented
+ How does hardware really multiply or divide numbers?

## 3.2 Addition and Subtraction

Addition is just what you would expect in computers. Digits are
added bit by bit from right to left. Subtraction uses addition: the appropriate operand is simply negated before added.

## 3.3 Multiplication

If we ignore the sign bits, the length of the multiplication of an $n$-bit
multiplicand and an $m$-bit multiplier is a product that is
$n + m$ bits long.

In this example, we restricted the decimal digits to 0 and 1. With only
two choices, each step of the multiplication is simple:

+ Just place a copy of the multiplicand in the proper place if the multiplier
digit is a 1.
+ Place 0 in the proper place if the digit is 0.

For now, let's assume that we are multiplying only positive numbers.

### 3.3.1 Sequential Version of the Multiplication Algorithm and hardware

Let's assume that the multiplier is in the 32-bit multiplier register
and that the 64-bit product register is initialized to 0


## 3.5 Floating Point

Going beyond signed and unsigned integers, programming languages support
numbers with fractions, which are called *reals* in mathematics.

Just as we can show decimal numbers in scientific notation, we can
also show binary numbers in scientific notation:

$$
1.0_{two} \times 2^{-1}
$$

Computer arithmetic that supports such numbers is called *floating point*
because it represents numbers in which the binary point is not
fixed, as it is for integers.

$$
1.xxxxxxxxx_{two} \times 2^{yyyy}
$$

### 3.5.1 Floating-Point Representation

A designer of a floating-point representation must find a compromise between
the size of the *fraction* and the size of the *exponent*, because
a fixed word size means you must take a bit from one to give a bit to the other.
This tradeoff is between *precision* and *range*: increasing the size of
the fraction enhances the precision of the fraction, while increasing
the size of the exponent increases the range of numbers that
can be represented.

The representation of a RISC-V floating-point number is shown below,
where *s* is the sign of the floating-point number, *exponent* is the
value of the 8-bit exponent field, and *fraction* is the 23-bit number.
This representation is *sign and magnitude*.

![RISC-V float](https://s2.loli.net/2022/04/18/2CJxNPshpfYyz7O.png)

In general, floating-point numbers are of the form

$$
(-1)^{S} \times F \times 2^{E}
$$

These chosen sizes of exponent and fraction give RISC-V computer arithmetic
an extraordinary range. Overflow exceptions can occur in floating-point arithmetic
as well as in integer arithmetic. Notice that overflow here means that
the exponent is too large to be represented in the exponent field.

Floating point offers a new kind of exceptional events as well. Just as
programmers will want to know when they have calculated a number
that is too large to be represented; either event could result in a program
giving incorrect answers. To distinguish it from overflows, we call
this event *underflow*. This situation occurs when the negative exponent
is too large to fit in the exponent field.

One way to reduce the chances of underflow or overflow is to offer another
format that has a larger exponent. In C, this number is called *double*,
and operations on doubles are called *double precision* floating-point
arithmetic.

The representation of a double precision floating-point number takes
one RISC-V doubleword, as shown below.

![RISC-V double](https://s2.loli.net/2022/04/18/6IpDOFMnXcBZGhT.png)

### 3.5.2 Exceptions and Interrupts

What should happen on an overflow or underflow to let the user know
that a problem occurred? Some computers signal these events by raising an *exception*,
sometimes called an *interruption*. RISC-V computers do not raise
an exception on overflow or underflow; instead, software can read
the *floating-point control and status register*(fcsr) to check
whether overflow or underflow has occurred.

### 3.5.3 IEEE 754 Floating-Point Standard

These formats go beyond RISC-V. They are part of the IEEE 754 floating-point standard,
found in virtually every computer invented since 1980.

To pack even more bits into the number, IEEE 754 makes the leading 1 bit
of normalized binary implicit. Hence, the number is actually 24 bits long in
single precision, and 53 bits long in double precision. To be precise,
we use the term *significand* to represent the 24- or 53-bit number that
is 1 plus the fraction. Since 0 has no leading 1, it is given the reserved
exponent value 0 so
that the hardware won't attach a leading 1 to it.

Thus $00 \dots 00_{two}$ represents 0; the representation of the rest
of the numbers uses the form from before with the hidden 1 added:

$$
(-1)^{S} \times (1 + Fraction) \times 2^{E}
$$

The designers for IEEE 754 also wanted a floating-point representation
that could be easily processed by integer comparisons, especially for sorting.
This desire is why the sign is in the most significant bit.

Placing the exponent before the significand also simplifies the sorting
of floating-point numbers using integer comparison instructions. Negative
exponents pose a challenge to simplified sorting. If we use two's complement
or any other notation in which negative exponents will look like a big number.

The desirable notation must therefore represent the most negative exponent as $00 \dots 00$
and the most positive as $11 \dots 11$. This convention is called *biased notation*,
with the bias being the number subtracted from the normal, unsigned
representation to determine the real value.

IEEE 754 uses a bias of 127 for single precision.

$$
(-1)^{S} \times (1 + Fraction) \times 2^{Exponent \quad Bias}
$$
