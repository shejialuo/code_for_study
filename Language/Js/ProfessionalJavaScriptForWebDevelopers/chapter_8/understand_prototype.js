'use strict'

function Person() {}

console.log(typeof Person.prototype);
console.log(Person.prototype);

console.log(Person.prototype.constructor === Person);

console.log(Person.prototype.__proto__ === Object.prototype);
console.log(Person.prototype.__proto__.constructor === Object);
console.log(Person.prototype.__proto__.__proto__ === null);

let person1 = new Person();
let person2 = new Person();

console.log(person1.__proto__ === Person.prototype);
