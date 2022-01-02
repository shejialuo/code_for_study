#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

/*
  * The SVR2 implementation of `sleep` used `setjmp` 
  * and `longjmp` to avoid the race condition.

  * The `sleep2` function avoids the race condition. Even if the `pause` is never executed,
  * the `sleep2` function returns when the `SIGALRM` occurs. However, if the `SIGALRM`
  * interrupts some other signal handler, then when we call `longjmp`, we abort the
  * other signal handler.
*/
static jmp_buf env_alrm;

static void sig_alrm(int signo) {
  longjmp(env_alrm, 1);
}

unsigned int sleep2(unsigned int seconds) {
  if (signal(SIGALRM, sig_alrm) == SIG_ERR)
    return seconds;
  if (setjmp(env_alrm) == 0) {
    alarm(seconds);
    pause();
  }
  return alarm(0);
}

static void sig_int(int signo) {
  int i,j;
  volatile int k;

  printf("\nsig_int staring\n");
  for(i = 0; i < 300000; ++i)
    for(j = 0; j < 4000; ++j)
      k += i * j;
  printf("sig_int finished\n");
}

int main(int argc, char *argv[]) {
  unsigned int unslept;

  if(signal(SIGINT, sig_int) == SIG_ERR)
    printf("error");
  unslept = sleep2(5);
  printf("sleep2 returned: %u\n", unslept);
  exit(0);
}

/*
  * We can see that the `longjmp` from the `sleep2` function aborted
  * the other signal
  * handler, `sig_int`, even though it wasn't finished.
*/