#include "tabtenn1.h"
#include <iostream>

TableTennisPlayer::TableTennisPlayer(const string& fn,
  const string& ln, bool ht): firstName(fn),
    lastName(ln),hasTable(ht) {}

void TableTennisPlayer::name() const {
  std::cout << lastName << ", " << firstName;
}

RatedPlayer::RatedPlayer(unsigned int r, const string& fn,
  const string&ln, bool ht): TableTennisPlayer(fn, ln ,ht) {
  rating = r;
}

RatedPlayer::RatedPlayer(unsigned int r, const TableTennisPlayer & tp)
  : TableTennisPlayer(tp), rating(r) {}

/*
  * A derived class has some special relationships
  * with the base class.
  * 1. A derived-class object can use base-class methods,
  *    provided that the methods are not private.
  ! RatedPlayer rplayer1(1140, "Mallory", "Duck", true);
  ! rplayer1.Name();
  * 2. A base class pointer can point to a derived class
  *    object without an explicit type cast and that a
  *    base-class reference can refer to a derived-class
  *    object without an explicit type cast.
  ! RatedPlayer rplayer1(1140, "Mallory", "Duck", true);
  ! TableTennisPlayer& rt = rplayer;
  ! TableTennisPlayer* pt = &rplayer;
  ! rt.Name();
  ! pt->Name();
  * However, a base-class pointer or reference can invoke
  * just base-class method.
*/

