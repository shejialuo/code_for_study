#include <iostream>
#include <string>
#include <set>
#include <algorithm>
using namespace std;

class Person {
public:
  string firstname() const;
  string lastname() const;
};

int main() {

  auto personSortCriterion = [](const Person& p1, const Person& p2) {
    return p1.lastname() < p2.lastname() ||
           (p1.lastname() == p2.lastname() &&
            p1.firstname() < p2.firstname());
  };

  /*
    * `personSortCriterion` is a function, however, `set` needs
    * a type so we use `decltype` and pass it to the constructor
  */
  set<Person, decltype(personSortCriterion)> coll(personSortCriterion);
  return 0;
}
