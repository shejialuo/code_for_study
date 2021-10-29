#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

vector<int> primeNumber;

void initPrimeNumber() {
  bool isPrime;
  for(int i = 2; i <= 10000; ++i) {
    isPrime = true;
    for(int j = 2; j <= sqrt(double(i)); ++j) {
      if (i % j == 0) {
        isPrime = false;
        break;
      }
    }
    if(isPrime) {
      primeNumber.push_back(i);
    }
  }
}

int sumOfConsecutivePrimeNumbers(const int number) {
  int pFirst = 0, pLast = 0, end = primeNumber.size();
  int ans = 0;
  int sum = 0;
  while (pLast < end) {
    if (sum == number) ans++;
    if (sum > number) {
      sum += -primeNumber[pFirst++];
      continue;
    }
      sum += primeNumber[pLast++];
  }
  return ans;
}

int main() {
  initPrimeNumber();
  int numberInput = 0;
  int ans = 0;
  while((cin >> numberInput) && numberInput != 0) {
    ans = sumOfConsecutivePrimeNumbers(numberInput);
    cout << ans << "\n";
  }
  return 0;
}