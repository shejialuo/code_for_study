// @copyright
// From https://medium.com/@ankur_anand/implement-your-own-call-apply-and-bind-method-in-javascript-42cc85dba1b

'use strict'

/**
 * @brief an example for demonstrating call
 * @param  message
 */
function showProfileMessage(message) {
  console.log(message, this.name)
}

const obj = {
  name: "Ankur Anand"
};

showProfileMessage.call(obj, "welcome")


// We can notice three things:
// 1. Calling the prototype function call changes the pointing of this.
//    Function call in the above became `obj.showProfileMessage`.
// 2. Whatever arguments we have passed should be passed to original
//    `showProfileMessage` as `arg1, arg2, ...`
// 3. Does not cause side effects to `obj` and `showProfileMessage` Function.
//    calling `call` doesn't modify the original `obj` or `showProfileMessage`

Function.prototype.myOwnCall = function(someOtherThis) {
  // when the function calls, we stores the function object
  // into `fnName
  someOtherThis = someOtherThis || global

  // We need to generate a random name
  let uniqueID = "00" + Math.random();
  while (someOtherThis.hasOwnProperty(uniqueID)) {
    uniqueID = "00" + Math.random();
  }
  let args = [];
  someOtherThis[uniqueID] = this

  for (var i = 1, len = arguments.length; i < len; i++) {
    args.push("arguments[" + i + "]");
  }

  // We use `eval` to pass the parameters.
  var result = eval("someOtherThis[uniqueID](" + args + ")");
  delete someOtherThis[uniqueID];
  return result;
}

showProfileMessage.myOwnCall(obj, "welcome")
