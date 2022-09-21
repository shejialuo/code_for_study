# Chapter 8 Objects, Classes, and Object-Oriented Programming

Objects are just hash tables: nothing more than a grouping of
name-value pairs where value may be data or a function.

The canonical way of creating a custom object is to create a
new instance of `Object` and add properties and methods to it.

[canonical_way_constructor](./canonical_way_constructor.js)

It can also be written with object literals.

[object_literal](./object_literal.js)

## 8.2 Object Creation

### 8.2.1 The Factory Pattern

The factory pattern is a well-known design pattern used in
SE to abstract away the process of creating specific objects.

[factory_object_create](./factory_object_create.js)

However, the factor pattern didn't address the issue of *object identification*.

### 8.2.2 The Function Constructor Pattern

[constructor_object_create](./constructor_object_create.js)

Calling a constructor in this manner will do the following:

+ A new object is created in memory.
+ The new object's internal `[[Prototype]]` pointer is assigned
to the constructor's `prototype` property.
+ The `this` value of the constructor is assigned to the new object.
+ The code inside the constructor is executed.

The only difference between constructor functions and other functions
is the way in which they are called.

```js
// use as a constructor
let person = new Person("Nicholas", 29, "Software Engineer");
person.sayName();   // "Nicholas"

// call as a function
Person("Greg", 27, "Doctor");  // adds to window
window.sayName();   // "Greg"

// call in the scope of another object
let o = new Object();
Person.call(o, "Kristen", 25, "Nurse");
o.sayName();  // "Kristen"
```

The major downside to constructors is that methods are created
once for each instance. So we could use another way.

```js
function Person(name, age, job){
  this.name = name;
  this.age = age;
  this.job = job;
  this.sayName = sayName;
}

function sayName() {
  console.log(this.name);
}

let person1 = new Person("Nicholas", 29, "Software Engineer");
let person2 = new Person("Greg", 27, "Doctor");

person1.sayName();
person2.sayName();
```

### 8.2.3 The Prototype Pattern

Each function is created with a `prototype` property, which is an object
containing properties and methods that should be available to
instances of a particular reference type. This object is literally
a prototype for the object to be created once the constructor is
called. The benefit of using the prototype is that all of its properties
and methods are shared among object instances.

[prototype_example](./prototype_example.js)

#### How Prototypes Work

Whenever a function is created, its `prototype` property is also
created according to a specific set of rules. By default, all
prototypes automatically get a property called `constructor` that
points back to the function on which it is a property.

When defining a custom constructor, the prototype gets the
`constructor` property only be default; all other methods are
inherited from `Object`. Each time the constructor is called to create
a new instance, that instance has an internal pointer to the constructor's
prototype, which is called `[[Prototype]]`. There is no standard
way to access `[[Prototype]]`, but the web browsers all support
a property on every object called `__proto__`.

[understand_prototype](./understand_prototype.js)

![Prototype of Person](https://s2.loli.net/2022/09/21/sdrC23vaBDSuj5L.png)

#### Understanding the Prototype Hierarchy

Whenever a property is accessed for reading on an objet, a search
is started to find a property with that name. The search begins
on object instance itself. If a property with the given name
is found on the instance, then that value is returned; if the
property is not found, then the search continues up the pointer
to the prototype, and the prototype is searched for a property
with the same name. If the property is found on the prototype,
then that value is returned.

```js
function Person() {}

Person.prototype.name = "Nicholas";
Person.prototype.age = 29;
Person.prototype.sayName = function() {
  console.log(this.name);
};
let person1 = new Person();
let person2 = new Person();

person1.name = "Greg";
console.log(person1.name);   // from instance
console.log(person2.name);   // from prototype


delete person1.name;
console.log(person1.name);   // from the prototype
```

![The relationship between instance and prototype](https://s2.loli.net/2022/09/21/BW97tjIru68GpUl.png)

#### Prototypes and the "in" operator

There are tow ways to use the `in` operator: on its own or
as a `for-in` loop. When used on its own, the `in` operator returns
`true` when a property of the given name is accessible by the
object, which is to say that the property may exist on the instance
or on the prototype.

When using a `for-in` loop, all properties that are accessible by
the object and can be enumerated will be returned, which includes
properties both on the instance and prototype.

## 8.3 Inheritance

### 8.3.1 Prototype Chaining

```js
function SuperType() {
  this.property = true;
}


SuperType.prototype.getSuperValue = function() {
  return this.property;
};

function SubType() {
  this.subproperty = false;
}

SubType.prototype = new SuperType();

SubType.prototype.getSubValue = function () {
  return this.subproperty;
};

let instance = new SubType();
console.log(instance.getSuperValue());
```

#### Default Prototypes

All reference types inherit from `Object` bye default.

#### Working with Methods

Often a subtype need to either override a supertype method or
introduce a new methods that don't exist on the supertype.
To accomplish this, the methods must be added to the prototype
after the prototype has been assigned.

### 8.3.2 Constructor Stealing

```js
function SuperType() {
  this.colors = ["red", "blue", "green"];
}

function SubType() {
  // inherit from SuperType
  SuperType.call(this);
}

let instance1 = new SubType();
instance1.colors.push("black");
console.log(instance1.colors);  // "red,blue,green,black"

let instance2 = new SubType();
console.log(instance2.colors);  // "red,blue,green"
```

## 8.4 Classes

### 8.4.1 Class Definition Basics

```js
class Person {}

const Animal = class {};
```

### 8.4.2 The Class Constructor

The `constructor` keyword is used inside the class definition block
to signal the definition of the class's constructor function.
Using the method name `constructor` will signal to the interpreter
that this particular function should be invoked to be a fresh instance using
the `new` operator.

+ A new object is created in memory.
+ The new object's internal `[[Prototype]]` pointer is assigned
to be the constructor's `prototype` property.
+ The `this` value of the constructor is assigned to the new object.
+ The code inside the constructor is executed.
