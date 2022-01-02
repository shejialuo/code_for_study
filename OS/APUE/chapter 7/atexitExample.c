#include <stdlib.h>
#include <stdio.h>

static void my_exit1();
static void my_exit2();

int main() {
  atexit(my_exit2);
  atexit(my_exit1);
  atexit(my_exit1);
  printf("main is done\n");
  exit(0);
}

static void my_exit1() {
  printf("first exit handler\n");
}

static void my_exit2() {
  printf("second exit handler\n");
}