#include <type_traits>
#include <utility>

template <class From, class To>
concept ConvertibleNoNarrowing = std::convertible_to<From, To>
    && requires(void (*foo)(To), From f) {
        foo({f});
};

auto foo_ni(ConvertibleNoNarrowing<int> auto ... args) {}

foo_ni(24, 12); // OK
foo_ni(24, (short)12); // OK
foo_ni(24, (long)12); // error
foo_ni(24, 12, 15.2); // error
