#include <algorithm>
#include <thread>
#include <vector>
using namespace std;

template<typename Iterator, typename T>
class AccumulateBlock {
public:
  void operator()(Iterator first, Iterator last, T& result) {
    result = accumulate(first, last, result);
  }
};

template<typename Iterator, typename T>
T ParallelAccumulate(Iterator first, Iterator last, T init) {
  const unsigned long length = distance(first, last);
  if(!length)
    return init;

  const unsigned long minPerThread = 25;
  const unsigned long maxThread =
    (length + minPerThread - 1) / minPerThread;

  const unsigned long hardwareThreads = thread::hardware_concurrency();

  const unsigned long numThreads = min
    (hardwareThreads !=0 ? hardwareThreads:2, maxThread);

  const unsigned long blockSize = length / numThreads;

  vector<T> results(numThreads);
  vector<thread> threads(numThreads - 1);

  Iterator blockStart = first;
  for(unsigned long i = 0; i < (numThreads - 1); ++i) {
    Iterator blockEnd = blockStart;
    advance(blockEnd, blockSize);
    threads[i] = thread(
      AccumulateBlock<Iterator, T>(),
      blockStart, blockEnd, ref(results[i]);
    )
    blockStart = blockEnd;
  }

  AccumulateBlock<Iterator, T>() (blockStart, last,
    results[numThreads - 1]);

  for_each(threads.begin(), threads.end(), mem_fn(thread::join));

  return accumulate(results.begin(), results.end(), init);
}
