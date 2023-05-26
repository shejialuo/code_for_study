class Collection {
public:
  template <typename T>  // an in-class member class template function
  class Node {

  };

  template <typename T>  // an in-class (and therefore implicitly inline)
  T* alloc() {           // member function template definition
    return new T;
  }

  template <typename T>  // a member variable template (since )
  static T zero = 0;

  template <typename T>  // a member alias template
  using NodePtr = Node<T*>;
};