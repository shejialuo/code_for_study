#include <iostream>
#include "tabtenn0.h"

TableTennisPlayer::TableTennisPlayer(const string& fn,
  const string& ln, bool ht): firstName(fn),
    lastName(ln),hasTable(ht) {}

void TableTennisPlayer::name() const {
  std::cout << lastName << ", " << firstName;
}
