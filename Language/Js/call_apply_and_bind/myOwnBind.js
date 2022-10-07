Function.prototype.myOwnBind = function(newThis) {
  if (typeof this !== "function") {
    throw new Error(this + "cannot be bound as it's not callable");
  }
  var boundTargetFunction = this;
  var boundArguments = Array.prototype.slice.call(arguments, 1);
  return function boundFunction() {
    // here the arguments refer to the second time when we call the target function returned from bind
    var targetArguments = Array.prototype.slice.call(arguments);
    return boundTargetFunction.apply(
      newThis,
      boundArguments.concat(targetArguments)
    );
  };
};
