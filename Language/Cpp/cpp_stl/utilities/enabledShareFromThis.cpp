#include <iostream>
#include <string>
#include <vector>
#include <memory>
using namespace std;

/*
  * `weak_ptr` allows sharing but not owning an object.
  * This class requires a shared pointer to get created.
  * Whenever the last shared pointer owning the object loses
  * its ownership, any weak pointer automatically becomes
  * empty.
  *
  * You can't use operator * and -> to access a referenced
  * object of a `weak_ptr` directly. Instead, you have to
  * create a shared pointer out of it.
*/
class Person {
public:
  string name;
  shared_ptr<Person> mother;
  shared_ptr<Person> father;
  vector<weak_ptr<Person>> kids;

  Person(const string & n,
         shared_ptr<Person> m = nullptr,
         shared_ptr<Person> f = nullptr)
    : name(n), mother(m), father(f) {}
  virtual ~Person() {
    cout  << "delete " << name << endl;
  }

  void setParentsAndTheirKidsWrong(shared_ptr<Person> m = nullptr,
                              shared_ptr<Person> f = nullptr) {
    mother = m;
    father = f;
    if (m != nullptr) {
      /*
        * Error, this is managed by another shared pointer
      */
      m ->kids.push_back(shared_ptr<Person>(this));
    }
    if (f != nullptr) {
      /*
        * Error, this is managed by another shared pointer
      */
      f->kids.push_back(shared_ptr<Person>(this));
    }
  }

  /*
    * One way to deal with this problem is to pass the shared
    * pointe to the kid as a third argument.
  */
  void setParentsAndTheirKidsCorrect(shared_ptr<Person> m = nullptr,
                                     shared_ptr<Person> f = nullptr,
                                     shared_ptr<Person> k = nullptr) {
    mother = m;
    father = f;
    if (m != nullptr) {
      m ->kids.push_back(k);
    }
    if (f != nullptr) {
      f->kids.push_back(k);
    }
  }
};


class PersonEnabledShareFromThisExample:
  public std::enable_shared_from_this<PersonEnabledShareFromThisExample> {
public:
  string name;
  shared_ptr<PersonEnabledShareFromThisExample> mother;
  shared_ptr<PersonEnabledShareFromThisExample> father;
  vector<weak_ptr<PersonEnabledShareFromThisExample>> kids;

  PersonEnabledShareFromThisExample(const string & n,
         shared_ptr<PersonEnabledShareFromThisExample> m = nullptr,
         shared_ptr<PersonEnabledShareFromThisExample> f = nullptr)
    : name(n), mother(m), father(f) {}
  virtual ~PersonEnabledShareFromThisExample() {
    cout  << "delete " << name << endl;
  }

  void setParentsAndTheirKidsWrong(shared_ptr<PersonEnabledShareFromThisExample> m = nullptr,
                              shared_ptr<PersonEnabledShareFromThisExample> f = nullptr) {
    mother = m;
    father = f;
    if (m != nullptr) {
      m ->kids.push_back(shared_from_this());
    }
    if (f != nullptr) {

      f->kids.push_back(shared_from_this());
    }
  }
};
