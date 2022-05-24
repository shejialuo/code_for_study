#include <iostream>
#include <string>
#include <vector>
#include <memory>
using namespace std;

/*
  * The major reson to use `shared_ptr` is to avoid taking
  * care of the resources a pointer refers to.
  *
  * However, under certain circumstances, this behavior doesn't
  * work or is not waht is intended:
  *
  * + One example is cyclic references. If two objects refer to
  *   each other using `shared_ptr`, and you want to release
  *   the objects and their associated resource if no other
  *   references to these objects exist, `shared_ptr` won't
  *   release the data, because the `use_count()` of each object
  *   is still 1.
  * + Another example occurs when you explicitly want to share
  *   but not own an object. Thus, you have the semantics that
  *   the lifetime of a reference to an object outlives the
  *   object it refers to
*/

/*
  * It's bad to use vector<shared_ptr<Person>>
*/
class Person {
public:
  string name;
  shared_ptr<Person> mother;
  shared_ptr<Person> father;
  vector<shared_ptr<Person>> kids;

  Person(const string & n,
         shared_ptr<Person> m = nullptr,
         shared_ptr<Person> f = nullptr)
    : name(n), mother(m), father(f) {}
  virtual ~Person() {
    cout  << "delete " << name << endl;
  }
};

shared_ptr<Person> initFamily(const string& name) {
  shared_ptr<Person> mom(new Person(name+ "'s mon"));
  shared_ptr<Person> dad(new Person(name +"'s dad"));
  shared_ptr<Person> kid(new Person(name, mom, dad));
  mom->kids.push_back(kid);
  dad->kids.push_back(kid);
  return kid;
}

int main() {
  shared_ptr<Person> p = initFamily("nico");

  cout << "nico's family exists" << endl;
  cout << "- nico is shared " << p.use_count() << " times" << endl;
  cout << "- name of 1st kid of nico's mon: "
       << p->mother->kids[0]->name << endl;

  // desctructor never happens
  p = initFamily("jim");
  cout << "jim's family exists" << endl;

  return 0;
}
