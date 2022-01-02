#include <signal.h>
#include <unistd.h>

/*
  * This function looks like the `sleep` function. 
  * but this implementation has three problems.

    * If the caller already has an alarm set, 
    * that alarm is erased by the first call to `alarm`.
    * We have modified the disposition for `SIGALRM` but we don't reset the disposition.
    * There is a race condition between the first call to `alarm` and the call to `pause`.
*/
static void sig_alrm(int signo) {}

unsigned int sleep1(unsigned int seconds) {
  if (signal(SIGALRM, sig_alrm) == SIG_ERR)
    return seconds;
  alarm(seconds);
  pause();
  return(alarm(0));
}