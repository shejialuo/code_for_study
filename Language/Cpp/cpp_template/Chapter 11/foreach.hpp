template<typename Iter, typename Callable>
void foreach(Iter current, Iter end, Callable op) {
  while (current != end) {
    op(*current);
    current++;
  }
}
