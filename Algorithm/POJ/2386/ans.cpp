#include <iostream>
using namespace std;

const int maxN = 100;
const int maxM = 100;

int N = 0;
int M = 0;

char field[maxN][maxM];

void dfs(int x, int y) {
  field[x][y] = '.';

  for(int dx = -1; dx <= 1; ++dx) {
    for(int dy= -1; dy <= 1; ++dy) {
      int nx = x + dx;
      int ny = y + dy;
      if(0 <= nx && nx < N && 0 <= ny && ny < M && field[nx][ny] == 'W')
        dfs(nx, ny);
    }
  }
}

int main() {

  int ans = 0;

  cin >> N >> M;

  for(int i = 0 ; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
      cin >> field[i][j];
    }
  }

  for(int i = 0 ; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
      if(field[i][j] == 'W') {
        dfs(i,j);
        ans++;
      }
    }
  }

  cout << ans << "\n";

  return 0;
}