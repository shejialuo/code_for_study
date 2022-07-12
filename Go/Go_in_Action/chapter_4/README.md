# 4. Arrays, slices, and maps

## 4.1 Array internals and fundamentals

Understanding how array works will help you appreciate the elegance
and power that slices and maps provide.

### 4.1.1 Internals

An array in Go is a fixed-length data type that contains a contiguous block
of elements of the same type.

### 4.1.2 Declaring and initializing

```go
// Declare an integer array of five elements.
var array [5]int;

// Declare an integer array of five elements.
// Initialize each element with a specific value.
array := [5]int{10, 20, 30, 40, 50}

// Declare an integer array.
// Initialize each element with a specific value.
// Capacity is determined based on the number of values initialized.
array := [...]int{10, 20, 30, 40, 50}

// Declare an integer array of five elements.
// Initialize index 1 and 2 with specific values.
// The rest of the elements contain their zero value.
array := [5]int{1 : 10, 2: 20}
```

### 4.1.3 Working with arrays

To access an individual element, use the `[]` operator.

```go
array := [5]int{10, 20, 30, 40, 50}
array[2] = 35
```

You can have an array of pointers.

```go
array := [5]*int{0: new(int), 1: new(int)}

*array[0] = 10
*array[1] = 20
```

An array is a value in Go. This means you can use it in an assignment
operation.

```go
var array1 [5]string
array2 := [5]string{"Red", "Blue", "Green", "Yellow", "Pink"}
array1 = array2
```

### 4.1.4 Multidimensional arrays

Arrays are always one-dimensional, but they can be composed to
create multidimensional arrays.

```go
  array := [4][2]int{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
  fmt.Println(array)
```

### 4.1.5 Passing arrays between functions

Just like C.

## 4.2 Slice internals and fundamentals

A *slice* is a data structure that provides a way for you to work with
and manage collections of data. Slices are built around the concept
of dynamic arrays that can grow and shrink as you see fit.

### 4.2.1 Creating and initializing

#### Make and slice literals

One way to create a slice is to use the built-in function `make`.
When you use `make`, one option you have is to specify the length
of the slice.

```go
slice := make([]string, 5)
```

When you just specify the length, the capacity of the slice is teh same.
You can also specify the length and capacity separately.

```go
slice := make([]int, 3, 5)
```

An idiomatic way of creating a slice is to use a slice literal.
It's similar to creating an array, except you don't specify a value
inside of the `[]` operator. The initial length and capacity will
be based on the number of elements you initialize.

```go
slice := []string{"Red", "Blue", "Green", "Yellow", "Pink"}
```

When using a slice literal, you can set the initial length and
capacity. All you need to do is initialize the index that represents the
length and capacity you need.

```go
slice := []string{99: ""}
```

#### NIL and empty slices

Sometimes in your programs you may need to declare a `nil` slice.
A `nil` slice is created by declaring a slice without any initialization.

```go
var slice []int
```

You can also create an empty slice of integers.

```go
slice1 := make([]int, 0)
slice2 := []int{}
```

### 4.2.3 Working with slices

#### Assigning and slicing

To change the value of an individual element, use the `[]` operator.

```go
slice := []int{10, 20, 30, 40, 50}
slice[1] = 25
```

Slices are called such because you can slice a portion of the
underlying array to create a new slice.

```go
slice := []int{10, 20, 30, 40, 50}
newSlice := slice[1:3]
```

After the slicing operation, we have two slices that are sharing the same
underlying array. However, each slice views the underlying array
in a different way. (See below)

![Two slices sharing the same underlying array](https://s2.loli.net/2022/07/12/mStTOedcI2blXai.png)

Calculating the length and capacity for any new slice is performed
using the following formula. For `slice[i:j]` of capacity `k`.

$$
\begin{align*}
Length &: j - i \\
Capacity &: k - i
\end{align*}
$$

#### Growing slices

One of the advantages of using a slice over using an array is
that you can grow the capacity of your slice as needed. Go
takes care of all the operational details when you use the built-in
function `append`.

When your `append` returns, it provides you a new slice with the changes.
The `append` function will always increase the length of the new slice.
The capacity, on the other hand, may or may not be affected,
depending on the available capacity of the source slice.

```go
slice := []int{10, 20, 30, 40, 50}
newSlice := slice[1:3]

newSlice = append(newSlice, 60)
```

After the `append` operation, the slices and the underlying array
will look like below.

![The underlying array after the append operation](https://s2.loli.net/2022/07/12/O2Kf5QgVRsJF9b1.png)

Because there was available capacity in the underlying array for
`newSlice`, the `append` operation incorporated the available element into
the slice's length and assigned the value.

When there's no available capacity in the underlying array for
a slice, the `append` function will create a new underlying array,
copy the existing values that are being referenced, and assign the new value.

```go
slice := []int{10, 20, 30, 40}
newSlice := append(slice, 50)
```

![The new underlying array after the append operation](https://s2.loli.net/2022/07/12/Cg1h9IQ6SpWiNTL.png)

#### Three index slices

There's a third index option which gives you control over the capacity
of the new slice. The purpose is to restrict the capacity.

```go
source := []string{"Apple", "Orange", "Plum", "Banana", "Grape"}
```

```go
// Slice the third element and restrict the capacity.
// Contains a length of 1 element and capacity of 2 elements.
slice := source[2:3:4]
```

For `slice[i:j:k]`, we can have the following formula.

$$
\begin{align*}
Length &: j -i \\
Capacity &: k - i
\end{align*}
$$

#### Iteration over slices

Since a slice is a collection, you can iterate over the elements.

```go
slice := []int{10, 20, 30, 40}

for index, value := range slice {
  fmt.Printf("Index: %d Value: %d\n", index, value)
}
```

It's important to kow the `range` is making a copy of the value, not
returning a reference.

## 4.3 Map internals and fundamentals

A map is a data structure that provides you with an unordered collection
of key/value pairs.

### 4.3.1 Internals

Maps are collections, and you can iterate over them just like you
do with arrays and slices. But maps are *unordered* collections, and
there's no way to predict the order in which the key/value pairs
will be returned.

### 4.3.2 Creating and initializing

There are several ways you can create and initialize maps in Go.
You can use the built-in function `make`, or you can use a map literal

```go
dict1 := make(map[string]int)
dict2 := make(map[string]string{"Red": "#da1337", "Orange": "#e95a22"})
```

The map key can be a value from any built-in or struct type as
long as the value can be used in an expression with the `==` operator.
