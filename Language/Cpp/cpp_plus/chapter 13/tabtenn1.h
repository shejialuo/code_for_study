#ifndef TABTENN1_H_
#define TABTENN1_H_

#include <string>
using std::string;

class TableTennisPlayer {
private:
  string firstName;
  string lastName;
  bool hasTable;
public:
  TableTennisPlayer(const string& fn = "none",
                    const string& ln = "none",
                    bool ht = false);
  void name() const;
  bool HasTable() const {return hasTable;};
  void resetTable(bool v) {hasTable = v;};
};

class RatedPlayer: public TableTennisPlayer {
private:
  unsigned int rating;
public:
  RatedPlayer(unsigned int r = 0, const string& fn = "none",
              const string& ln = "none", bool ht = false);
  RatedPlayer(unsigned int r, const TableTennisPlayer& tp);
  unsigned int Rating() const {return rating;}
  void resetRating(unsigned int r) {rating = r;}
};

#endif // TABTENN1_H_