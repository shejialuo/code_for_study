template <typename To, typename From>
To safe_interpret_cast(From from) {
  static_assert(sizeof(From) <= sizeof(To), "Destination type too narrow");
  return reinterpret_cast<To>(from);
}
